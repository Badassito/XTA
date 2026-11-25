"""Prepare or verify a reusable first Update Both action on a native review."""
from contextlib import ExitStack, closing
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import tempfile

import numpy as np

from .native_review_session import (
    _APPLICATION_ID,
    _SCHEMA_VERSION,
    _UPDATE_BOTH_VERSION,
    _unpack,
    _update_both_counts,
)

_PREPARED_ID = 0x58544E55
_PREPARED_VERSION = 1
_BLOCK_KEYS = "layer,cutoff,z,y,x"


def initial_update_path(store_path):
    return Path(store_path).with_suffix(".initial-update.sqlite")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _source_binding(store):
    catalogue = {
        "geometry": store.geometry,
        "cutoffs": store.cutoffs,
        "layers": store.layers,
        "global_layer": store.global_layer_id,
    }
    return {
        "source_identity": store.identity,
        "shape": list(store.shape),
        "block_shape": list(store.block_shape),
        "manifest_sha256": getattr(store, "review_manifest_sha256", None),
        "catalogue_sha256": hashlib.sha256(_canonical(catalogue)).hexdigest(),
        "algorithm": _UPDATE_BOTH_VERSION,
    }


def _block_extent(store, key):
    lid, cutoff, *block = key
    layer = store.layers.get(lid)
    if (layer is None or not (layer["additive"] or lid == store.global_layer_id)
            or type(cutoff) is not int or cutoff not in store.cutoffs
            or any(type(i) is not int or i < 0 or i * step >= size
                   for i, step, size in zip(block, store.block_shape, store.shape))):
        raise ValueError("Prepared Update Both contains a block outside the source review")
    return tuple(min(step, size - i * step) for i, step, size in zip(block, store.block_shape, store.shape))


def _digest_row(digest, key, payload, extent):
    bound = (math.prod(extent) + 7) // 8
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= bound + bound // 100 + 128:
        raise ValueError("Invalid prepared Update Both block length")
    address = _canonical(key)
    digest.update(struct.pack("<II", len(address), len(payload)))
    digest.update(address)
    digest.update(payload)


def _payload_limit(store):
    upper = (math.prod(store.block_shape) + 7) // 8
    return upper + upper // 100 + 128


def _settled_totals(db, position, store, version):
    if version < 3:
        return None
    row = db.execute("SELECT global_voxels FROM update_both_states WHERE position=? AND algorithm=?",
                     (position, _UPDATE_BOTH_VERSION)).fetchone()
    return None if row is None else _update_both_counts(json.loads(row[0]), store)


def _baseline_global_count(store, cutoff):
    segments = store.layers[store.global_layer_id].get("segments_by_cutoff")
    if segments is not None:
        row = segments.get(str(cutoff), segments.get(cutoff))
        return 0 if row is None else int(row["foreground_voxels"])
    return sum(int(np.count_nonzero(store.read_block(store.global_layer_id, cutoff, block)))
               for block in store.iter_blocks(store.global_layer_id, cutoff))


def _copy_journal_action(source, destination, store, *, recorded_counts, trusted_legacy):
    app = source.execute("PRAGMA application_id").fetchone()[0]
    version = source.execute("PRAGMA user_version").fetchone()[0]
    if app != _APPLICATION_ID or version not in (2, _SCHEMA_VERSION):
        raise ValueError("Expected a native-review edit journal")
    row = source.execute("SELECT source_id,position,stale,revision FROM state WHERE id=1").fetchone()
    if row is None or row[0] != store.identity or row[2] != 0 or row[1:] not in ((0, 0, 0), (1, 0, 1)):
        raise ValueError("The journal must contain only one pristine Update Both for this archive")
    position = row[1]
    settled = _settled_totals(source, position, store, version)
    if settled is None and (not trusted_legacy or version != 2 or not position):
        raise ValueError("The journal lacks a verified Update Both receipt. Recompute or explicitly import a trusted version-2 journal.")
    actions = source.execute("SELECT position,description,stale_before,stale_after FROM actions LIMIT 2").fetchall()
    if actions != ([(1, "Update Both", 0, 0)] if position else []):
        raise ValueError("The journal is not a single Update Both action")
    if (source.execute("SELECT 1 FROM changes WHERE before IS NOT NULL OR after IS NULL OR position<>1 LIMIT 1").fetchone()
            or source.execute("SELECT 1 FROM changes LIMIT 1").fetchone() is not None and not position):
        raise ValueError("The journal has changes outside the pristine Update Both action")
    for left, right in (("changes", "masks"), ("masks", "changes")):
        left_payload = "after" if left == "changes" else "payload"
        right_payload = "after" if right == "changes" else "payload"
        if source.execute(f"SELECT {_BLOCK_KEYS},{left_payload} FROM {left} EXCEPT "
                          f"SELECT {_BLOCK_KEYS},{right_payload} FROM {right} LIMIT 1").fetchone() is not None:
            raise ValueError("Journal masks differ from their recorded Update Both action")

    totals = {cutoff: _baseline_global_count(store, cutoff) for cutoff in store.cutoffs}
    changes = source.execute(f"SELECT {_BLOCK_KEYS},substr(after,1,?),length(after) "
                             f"FROM changes ORDER BY {_BLOCK_KEYS}", (_payload_limit(store),))
    rows = 0
    for row in changes:
        key, payload, payload_size = row[:-2], row[-2], row[-1]
        lid, cutoff, *index = key
        extent = _block_extent(store, key)
        if not isinstance(payload, bytes) or len(payload) != payload_size:
            raise ValueError("Invalid Update Both journal block length")
        new_mask = _unpack(payload, extent)
        old_mask = store.read_block(lid, cutoff, tuple(index))
        if np.array_equal(new_mask, old_mask):
            raise ValueError("Pristine Update Both action records an unchanged block")
        if lid == store.global_layer_id:
            totals[cutoff] += int(np.count_nonzero(new_mask)) - int(np.count_nonzero(old_mask))
        elif np.any(new_mask & ~old_mask):
            raise ValueError("Pristine Update Both adds foreground to a contributor")
        else:
            retained = source.execute(
                """SELECT substr(payload,1,?),length(payload) FROM masks
                WHERE layer=? AND cutoff=? AND z=? AND y=? AND x=?""",
                (_payload_limit(store), store.global_layer_id, cutoff, *index),
            ).fetchone()
            if retained is None:
                global_mask = store.read_block(store.global_layer_id, cutoff, tuple(index))
            else:
                if not isinstance(retained[0], bytes) or len(retained[0]) != retained[1]:
                    raise ValueError("Invalid Update Both Global block length")
                global_mask = _unpack(retained[0], extent)
            if not np.array_equal(new_mask, old_mask & global_mask):
                raise ValueError("Pristine Update Both changes a contributor outside its Global clip")
        destination.execute("INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?)", (*key, payload))
        rows += 1
    if position and not rows:
        raise ValueError("The legacy Update Both action contains no mask changes")
    totals = _update_both_counts(totals, store)
    if (settled is not None and totals != settled) or (recorded_counts is not None and totals != recorded_counts):
        raise ValueError("The prepared Update Both mask totals disagree with its receipt")
    return totals, rows


def _rows_digest(db, metadata, *, store, schema="main"):
    # Both metadata and text/blob fields are length-delimited. The SHA detects
    # accidental modification independently of the SQLite page layout.
    digest = hashlib.sha256(_canonical(metadata) + b"\n")
    rows = 0
    for row in db.execute(
        f"SELECT {_BLOCK_KEYS},substr(payload,1,?),length(payload) FROM {schema}.blocks ORDER BY {_BLOCK_KEYS}",
        (_payload_limit(store),),
    ):
        key, payload, payload_size = row[:-2], row[-2], row[-1]
        if not isinstance(payload, bytes) or len(payload) != payload_size:
            raise ValueError("Invalid prepared Update Both block length")
        _digest_row(digest, key, payload, _block_extent(store, key))
        rows += 1
    return digest.hexdigest(), rows


def verify_prepared_update(db, store):
    """Verify a compatible sidecar in the same pinned SQLite transaction used to apply it."""
    if (db.execute("PRAGMA prepared_update.application_id").fetchone()[0] != _PREPARED_ID
            or db.execute("PRAGMA prepared_update.user_version").fetchone()[0] != _PREPARED_VERSION):
        raise ValueError("The prepared Update Both file uses an unsupported schema")
    tables = db.execute("SELECT type,name FROM prepared_update.sqlite_schema "
                        "WHERE name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    if tables != [("table", "blocks"), ("table", "receipt")]:
        raise ValueError("Invalid prepared Update Both schema")
    rows = db.execute("SELECT length(document),substr(document,1,65537),digest "
                      "FROM prepared_update.receipt LIMIT 2").fetchall()
    if len(rows) != 1 or type(rows[0][0]) is not int or rows[0][0] > 65536:
        raise ValueError("The prepared Update Both receipt is missing or oversized")
    _, document, expected_digest = rows[0]
    try:
        receipt = json.loads(document)
        binding, counts, expected_count = receipt["source"], receipt["global_voxels"], receipt["block_count"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid prepared Update Both receipt") from error
    if binding != _source_binding(store):
        raise ValueError("Prepared Update Both belongs to a different native review or algorithm")
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("Invalid prepared Update Both block count")
    totals = _update_both_counts(counts, store)
    digest, count = _rows_digest(db, receipt, schema="prepared_update", store=store)
    if count != expected_count or digest != expected_digest:
        raise ValueError("Prepared Update Both failed its integrity check")
    return totals, count


def prepare_initial_update(store_or_path, journal_path, destination=None, *, global_voxels=None,
                           trusted_legacy=False):
    """Publish one validated, reusable action without modifying the archive or journal."""
    from .json_publication import _fsync_parent_directory
    from .native_review_store import NativeReviewStore
    from .run_transport import _publish_fresh

    with ExitStack() as resources:
        store = store_or_path
        if isinstance(store, (str, os.PathLike)):
            store = resources.enter_context(NativeReviewStore(store))
        if store.global_layer_id is None:
            raise ValueError("Native review has no Global layer")
        if destination is None:
            destination = initial_update_path(store.path)
        destination = Path(destination).expanduser().absolute()
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        counts = None if global_voxels is None else _update_both_counts(global_voxels, store)
        path = Path(journal_path).expanduser().resolve(strict=True)
        original = resources.enter_context(closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)))
        original.execute("PRAGMA trusted_schema=OFF")
        original.execute("PRAGMA query_only=ON")
        original.execute("PRAGMA temp_store=FILE")
        original.execute("BEGIN")
        destination.parent.mkdir(parents=True, exist_ok=True)
        directory = resources.enter_context(tempfile.TemporaryDirectory(prefix=".native-initial-update-", dir=destination.parent))
        stage = Path(directory) / "prepared.sqlite"
        with closing(sqlite3.connect(stage)) as prepared:
            prepared.executescript(f"""
                PRAGMA application_id={_PREPARED_ID};
                PRAGMA user_version={_PREPARED_VERSION};
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA cache_size=-8192;
                CREATE TABLE receipt(document TEXT NOT NULL, digest TEXT NOT NULL);
                CREATE TABLE blocks(
                    layer TEXT NOT NULL, cutoff INTEGER NOT NULL, z INTEGER NOT NULL,
                    y INTEGER NOT NULL, x INTEGER NOT NULL, payload BLOB NOT NULL,
                    PRIMARY KEY(layer,cutoff,z,y,x)
                ) WITHOUT ROWID;
            """)
            after, count = _copy_journal_action(original, prepared, store, recorded_counts=counts,
                                                trusted_legacy=trusted_legacy)
            receipt = {"source": _source_binding(store),
                       "global_voxels": {str(cutoff): n for cutoff, n in after.items()},
                       "block_count": count}
            digest, verified_count = _rows_digest(prepared, receipt, store=store)
            if verified_count != count:
                raise ValueError("Prepared Update Both block count changed during staging")
            prepared.execute("INSERT INTO receipt VALUES (?, ?)", (_canonical(receipt).decode("ascii"), digest))
            prepared.commit()
        with stage.open("rb+") as stream:
            os.fsync(stream.fileno())
        _publish_fresh(stage, destination)
        _fsync_parent_directory(destination.parent)
        return destination
