"""Durable native-grid edits and bounded export for a windowed review."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import heapq
import itertools
import json
import logging
import math
import operator
import os
from pathlib import Path
import sqlite3
import tempfile
import zlib

import numpy as np

from .native_review_store import _region_filter


_APPLICATION_ID = 0x5854524A
_SCHEMA_VERSION = 3
_UPDATE_BOTH_VERSION = "xta.native.update-both/1"
_PAINTED = -1
_MIB = 1024 * 1024
_BLOCK_LOOKUP = "layer=? AND cutoff=? AND z=? AND y=? AND x=?"

_TABLES = (
    """CREATE TABLE state (
        id INTEGER PRIMARY KEY CHECK(id=1), source_id TEXT NOT NULL,
        position INTEGER NOT NULL, stale INTEGER NOT NULL CHECK(stale IN (0, 1)),
        revision INTEGER NOT NULL)""",
    """CREATE TABLE actions (
        position INTEGER PRIMARY KEY, description TEXT NOT NULL,
        stale_before INTEGER NOT NULL, stale_after INTEGER NOT NULL)""",
    """CREATE TABLE changes (
        position INTEGER NOT NULL REFERENCES actions(position) ON DELETE CASCADE,
        layer TEXT NOT NULL, cutoff INTEGER NOT NULL,
        z INTEGER NOT NULL, y INTEGER NOT NULL, x INTEGER NOT NULL,
        before BLOB, after BLOB,
        PRIMARY KEY(position, layer, cutoff, z, y, x)) WITHOUT ROWID""",
    """CREATE TABLE masks (
        layer TEXT NOT NULL, cutoff INTEGER NOT NULL,
        z INTEGER NOT NULL, y INTEGER NOT NULL, x INTEGER NOT NULL,
        payload BLOB NOT NULL,
        PRIMARY KEY(layer, cutoff, z, y, x)) WITHOUT ROWID""",
    "CREATE INDEX mask_location ON masks(z, y, x, cutoff, layer)",
    """CREATE TABLE update_both_states (
        position INTEGER PRIMARY KEY, algorithm TEXT NOT NULL, global_voxels TEXT NOT NULL)""",
)


def _coords(value, label):
    raw = tuple(value)
    if len(raw) != 3 or any(isinstance(item, (bool, np.bool_)) for item in raw):
        raise ValueError(f"{label} must contain three integer coordinates in ZYX order")
    try:
        return tuple(operator.index(item) for item in raw)
    except TypeError as error:
        raise ValueError(f"{label} must contain three integer coordinates in ZYX order") from error


def _pack(mask):
    return zlib.compress(np.packbits(mask, bitorder="little").tobytes(), level=1)


def _inflate_bits(payload, size):
    packed_bytes = (size + 7) // 8
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, packed_bytes + 1)
    except zlib.error as error:
        raise ValueError("Native-review journal contains a corrupt block") from error
    if len(raw) != packed_bytes or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Native-review journal contains an invalid block size")
    if size % 8 and raw[-1] >> (size % 8):
        raise ValueError("Native-review journal contains invalid padding bits")
    return np.frombuffer(raw, np.uint8)


def _unpack(payload, shape):
    size = math.prod(shape)
    return np.unpackbits(_inflate_bits(payload, size), count=size, bitorder="little").reshape(shape).view(bool)


class StaleReviewWindow(RuntimeError):
    """A gesture targeted a viewport loaded before an intervening edit."""


def _check_revision(expected, actual):
    if expected is not None and expected != actual:
        raise StaleReviewWindow("The native review changed since this window loaded. Refresh before editing.")


def _update_both_counts(value, store):
    if not isinstance(value, dict):
        raise ValueError("Invalid Update Both foreground totals")
    counts = {str(cutoff): count for cutoff, count in value.items()}
    largest = math.prod(store.shape)
    if (len(counts) != len(value) or counts.keys() != {str(cutoff) for cutoff in store.cutoffs}
            or any(type(n) is not int or not 0 <= n <= largest for n in counts.values())):
        raise ValueError("Invalid Update Both foreground totals")
    return {int(cutoff): counts[cutoff] for cutoff in counts}


class NativeReviewSession:
    """Overlay the immutable review with a separate SQLite edit history.

    Store and journal each have their own lifetime. A gesture stages only touchable
    blocks, sees its own writes, and commits the new masks and their inverse in
    one database transaction. A journal can be reopened after changing windows.
    """

    def __init__(self, store_or_path, journal_path=None, *, max_region_bytes=256 * _MIB,
                 initial_update_path=None):
        self._owns_store = isinstance(store_or_path, (str, os.PathLike))
        if self._owns_store:
            from .native_review_store import NativeReviewStore

            self.store = NativeReviewStore(store_or_path)
        else:
            self.store = store_or_path
        self._db = None
        self._editing = False
        self._stale = None
        self._settled_update = None
        try:
            self.shape = tuple(self.store.shape)
            self.block_shape = tuple(self.store.block_shape)
            self.chunk_shape = self.block_shape
            self.cutoffs = tuple(self.store.cutoffs)
            self.layers = self.store.layers
            self.global_layer_id = self.store.global_layer_id
            self.identity = str(self.store.identity)
            self._additive_ids = tuple(key for key, layer in self.layers.items() if layer["additive"])
            self.max_region_bytes = operator.index(max_region_bytes)
            if self.max_region_bytes < 1:
                raise ValueError("Native review region byte limit must be positive")
            store_path = Path(self.store.path).resolve()
            self._initial_update_explicit = initial_update_path is not None and initial_update_path is not False
            self._initial_update = (
                None if initial_update_path is False else
                Path(initial_update_path).expanduser().absolute() if self._initial_update_explicit else
                store_path.with_suffix(".initial-update.sqlite")
            )
            self.path = Path(journal_path) if journal_path is not None else store_path.with_suffix(".edits.sqlite")
            self.path = self.path.expanduser().absolute()
            if (self.path.resolve() == store_path
                    or (self.path.exists() and self.path.samefile(store_path))):
                raise ValueError("The edit journal must be separate from the native review store")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path.as_uri(), uri=True, timeout=20, isolation_level=None)
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA trusted_schema=OFF")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA cache_size=-8192")
            self._db.execute("PRAGMA temp_store=FILE")
            self._db.execute("PRAGMA temp.cache_size=-8192")
            self._initialize(self.identity)
        except BaseException:
            self.close()
            raise

    def _initialize(self, source_id):
        with self._database_transaction():
            app_id = self._db.execute("PRAGMA application_id").fetchone()[0]
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if app_id == 0 and version == 0 and not self._db.execute("SELECT 1 FROM sqlite_master").fetchone():
                for statement in _TABLES:
                    self._db.execute(statement)
                self._db.execute("INSERT INTO state VALUES (1, ?, 0, 0, 0)", (source_id,))
                self._db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                self._db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            elif app_id != _APPLICATION_ID or version not in (1, 2, _SCHEMA_VERSION):
                raise ValueError("Unrecognized native-review edit journal")
            elif self._db.execute("SELECT source_id FROM state WHERE id=1").fetchone() != (source_id,):
                raise ValueError("This edit journal belongs to a different native review store")
            elif version < _SCHEMA_VERSION:
                if version == 1:
                    self._db.execute("ALTER TABLE state ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
                self._db.execute(_TABLES[-1])
                self._db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        self._db.execute("""CREATE TEMP TABLE pending (
            layer TEXT NOT NULL, cutoff INTEGER NOT NULL, z INTEGER NOT NULL,
            y INTEGER NOT NULL, x INTEGER NOT NULL, before BLOB, after BLOB,
            PRIMARY KEY(layer, cutoff, z, y, x)) WITHOUT ROWID""")
        self._db.execute("CREATE INDEX temp.pending_location ON pending(z, y, x, cutoff, layer)")
        self._db.execute("""CREATE TEMP TABLE brush_blocks (
            z INTEGER NOT NULL, y INTEGER NOT NULL, x INTEGER NOT NULL,
            PRIMARY KEY(z, y, x)) WITHOUT ROWID""")

    @contextmanager
    def _database_transaction(self, *, write=True):
        if self._db is None:
            raise RuntimeError("The native review session is closed")
        if self._db.in_transaction:
            raise RuntimeError("Native review operations must run in separate transactions")
        self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise

    def close(self):
        if self._editing:
            raise RuntimeError("Finish the native review gesture before closing the session")
        try:
            if self._db is not None:
                self._db.close()
                self._db = None
        finally:
            if self._owns_store:
                self._owns_store = False
                self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    @property
    def global_stale(self):
        if self._editing:
            return self._stale
        return bool(self._db.execute("SELECT stale FROM state WHERE id=1").fetchone()[0])

    @property
    def revision(self):
        """Generation of the active edit state, including undo and redo."""
        return self._db.execute("SELECT revision FROM state WHERE id=1").fetchone()[0]

    @property
    def display_state(self):
        """Journal generation and active position for a pinned display read."""
        return self._db.execute("SELECT revision,position FROM state WHERE id=1").fetchone()

    @contextmanager
    def snapshot(self):
        """Pin a consistent set of block reads and visibility queries."""
        with self._database_transaction(write=False):
            yield self.revision

    def set_global_stale(self, flag):
        self._require_edit()
        self._stale = bool(flag)

    def _require_edit(self):
        if not self._editing:
            raise RuntimeError("Native mask changes require an edit transaction")

    @contextmanager
    def transaction(self, description, *, expected_revision=None):
        """Group writes, brush changes and Global state into one undoable edit."""
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Native review edits need an action description")
        with self._database_transaction():
            position, stale, revision = self._db.execute("SELECT position, stale, revision FROM state WHERE id=1").fetchone()
            _check_revision(expected_revision, revision)
            self._editing, self._stale = True, bool(stale)
            self._settled_update = None
            try:
                yield self
                self._follow_brush_global()
                self._commit_pending(position, stale, description)
            finally:
                self._editing, self._stale, self._settled_update = False, None, None

    def record_update_both(self, global_voxels):
        """Mark this successful Update Both result for a subsequent no-op."""
        self._require_edit()
        self._settled_update = _update_both_counts(global_voxels, self)

    def _write_settled_update(self, position, totals):
        self._db.execute(
            "INSERT OR REPLACE INTO update_both_states VALUES (?, ?, ?)",
            (position, _UPDATE_BOTH_VERSION, json.dumps(totals, sort_keys=True, separators=(",", ":"))),
        )

    def _read_settled_update(self, position):
        row = self._db.execute(
            "SELECT global_voxels FROM update_both_states WHERE position=? AND algorithm=?",
            (position, _UPDATE_BOTH_VERSION),
        ).fetchone()
        return None if row is None else _update_both_counts(json.loads(row[0]), self)

    def settled_update_ancestor(self):
        """Find the most recent known Update Both on this active history branch."""
        self._require_edit()
        row = self._db.execute(
            """SELECT position,global_voxels FROM update_both_states
            WHERE algorithm=? AND position<= (SELECT position FROM state WHERE id=1)
            ORDER BY position DESC LIMIT 1""", (_UPDATE_BOTH_VERSION,),
        ).fetchone()
        return None if row is None else (row[0], _update_both_counts(json.loads(row[1]), self))

    def _validate_ancestor(self, position):
        self._require_edit()
        if (type(position) is not int or position < 0
                or not self._db.execute(
                    """SELECT 1 FROM update_both_states WHERE position=? AND algorithm=?
                    AND position<= (SELECT position FROM state WHERE id=1)""",
                    (position, _UPDATE_BOTH_VERSION),
                ).fetchone()):
            raise ValueError("Expected a settled Update Both on the active journal branch")

    def iter_changed_blocks_since(self, position):
        """Yield changed blocks in journal order, including repeats across actions."""
        self._validate_ancestor(position)
        for lid, cutoff, z, y, x in self._db.execute(
            """SELECT layer,cutoff,z,y,x FROM changes
            WHERE position>? AND position<= (SELECT position FROM state WHERE id=1) AND cutoff>=0
            ORDER BY position,layer,cutoff,z,y,x""", (position,),
        ):
            yield lid, cutoff, (z, y, x)

    def read_historical_block(self, layer_id, cutoff, block_zyx, position):
        """Read a native block as of an active-branch Update Both receipt."""
        self._validate_ancestor(position)
        key = self._key(layer_id, cutoff, block_zyx)
        extent = self.block_extent(key[2:])
        size = (math.prod(extent) + 7) // 8
        limit = size + size // 100 + 128
        row = self._db.execute(
            f"""SELECT length(before),substr(before,1,?) FROM changes WHERE position>? AND {_BLOCK_LOOKUP}
            AND position<= (SELECT position FROM state WHERE id=1) ORDER BY position LIMIT 1""",
            (limit, position, *key),
        ).fetchone()
        if row is None:
            return self._read_block(key, include_pending=False)
        if row[0] is None:
            return self._base_block(key)
        if row[0] > limit:
            raise ValueError("Native-review journal contains an oversized historical block")
        return _unpack(row[1], extent)

    def _matches_initial_update(self, position, stale):
        """Compare the first journal action byte for byte with the verified sidecar."""
        db = self._db
        if db.execute("SELECT 1 FROM changes WHERE position=1 AND before IS NOT NULL LIMIT 1").fetchone():
            return False
        # Check both directions so missing and extra blocks are equally invalid.
        address = "p.layer=c.layer AND p.cutoff=c.cutoff AND p.z=c.z AND p.y=c.y AND p.x=c.x"
        match = f"{address} AND c.position=1 AND p.payload=c.after"
        if db.execute(f"""SELECT 1 FROM changes c WHERE c.position=1 AND NOT EXISTS
                (SELECT 1 FROM prepared_update.blocks p WHERE {match}) LIMIT 1""").fetchone():
            return False
        if db.execute(f"""SELECT 1 FROM prepared_update.blocks p WHERE NOT EXISTS
                (SELECT 1 FROM changes c WHERE {match}) LIMIT 1""").fetchone():
            return False
        if position != 1:
            return True
        # At position 1 the currently applied overrides must also match.
        match = f"{address} AND p.payload=c.payload"
        return (
            not stale
            and not db.execute(f"""SELECT 1 FROM masks c WHERE NOT EXISTS
                (SELECT 1 FROM prepared_update.blocks p WHERE {match}) LIMIT 1""").fetchone()
            and not db.execute(f"""SELECT 1 FROM prepared_update.blocks p WHERE NOT EXISTS
                (SELECT 1 FROM masks c WHERE {match}) LIMIT 1""").fetchone()
        )

    def try_update_both(self, *, expected_revision=None, progress=None):
        """Return a settled result or apply one verified pristine-source action."""
        def result(totals, source):
            return {"status": "applied", "global_voxels": totals,
                    "global_stale": False, "update_source": source}

        attached = False
        try:
            with self._database_transaction():
                position, stale, revision = self._db.execute(
                    "SELECT position, stale, revision FROM state WHERE id=1",
                ).fetchone()
                _check_revision(expected_revision, revision)
                settled = self._read_settled_update(position)
                if not stale and settled is not None:
                    return result(settled, "settled")

                first_action = self._db.execute(
                    "SELECT stale_before,stale_after FROM actions WHERE position=1 AND description='Update Both'",
                ).fetchone()
                if position:
                    # Older journals have no settled receipts. Verify their
                    # first action only when it can seed an otherwise unknown
                    # history. Later active edits and redo entries stay intact.
                    if (first_action != (0, 0) or self._db.execute(
                            "SELECT 1 FROM update_both_states WHERE algorithm=? AND position<=? LIMIT 1",
                            (_UPDATE_BOTH_VERSION, position),
                    ).fetchone()):
                        return None
                else:
                    if stale or self._db.execute("SELECT 1 FROM masks LIMIT 1").fetchone():
                        return None
                    settled = self._read_settled_update(1) if first_action == (0, 0) else None
                    if settled is not None:
                        # A prior native action can be applied again from
                        # pristine state even after the sidecar is moved.
                        self._db.execute("DELETE FROM actions WHERE position > 1")
                        self._db.execute("DELETE FROM update_both_states WHERE position > 1")
                        self._apply_changes(1, restore_before=False)
                        self._db.execute("UPDATE state SET position=1,revision=revision+1 WHERE id=1")
                        return result(settled, "recorded")

                path = self._initial_update
                if path is None or (not self._initial_update_explicit and not path.is_file()):
                    return None
                try:
                    from .native_review_initial import verify_prepared_update

                    if path.samefile(self.path) or path.samefile(self.store.path):
                        raise ValueError("The prepared Update Both must be a separate file")
                    self._db.execute("ATTACH DATABASE ? AS prepared_update", (path.resolve().as_uri() + "?mode=ro",))
                    attached = True
                    totals, count = verify_prepared_update(self._db, self.store)
                except (OSError, sqlite3.Error, ValueError) as error:
                    if self._initial_update_explicit:
                        raise
                    logging.getLogger(__name__).warning("Could not use prepared Update Both: %s", error)
                    return None
                if position:
                    if count and self._matches_initial_update(position, stale):
                        self._write_settled_update(1, totals)
                        if position == 1:
                            return result(totals, "settled")
                    return None
                if progress is not None:
                    progress("apply_prepared", None, 0)
                if count:
                    self._db.execute("DELETE FROM actions")
                    self._db.execute("DELETE FROM update_both_states WHERE position>0")
                    self._db.execute("INSERT INTO actions VALUES (1, 'Update Both', 0, 0)")
                    self._db.execute("""INSERT INTO changes
                        SELECT 1, layer, cutoff, z, y, x, NULL, payload FROM prepared_update.blocks""")
                    self._apply_changes(1, restore_before=False)
                    self._db.execute("UPDATE state SET position=1,revision=revision+1 WHERE id=1")
                self._write_settled_update(int(bool(count)), totals)
                if progress is not None:
                    progress("apply_prepared", None, count)
                return result(totals, "prepared")
        finally:
            if attached:
                self._db.execute("DETACH DATABASE prepared_update")

    def _commit_pending(self, position, stale_before, description):
        self._db.execute("DELETE FROM pending WHERE before IS after")
        if stale_before != self._stale or self._db.execute("SELECT 1 FROM pending LIMIT 1").fetchone():
            self._db.execute("DELETE FROM actions WHERE position > ?", (position,))
            self._db.execute("DELETE FROM update_both_states WHERE position > ?", (position,))
            position += 1
            self._db.execute("INSERT INTO actions VALUES (?, ?, ?, ?)", (position, description, stale_before, self._stale))
            self._db.execute("INSERT INTO changes SELECT ?, layer, cutoff, z, y, x, before, after FROM pending", (position,))
            self._apply_changes(position, restore_before=False)
            self._db.execute(
                "UPDATE state SET position=?, stale=?, revision=revision+1 WHERE id=1", (position, self._stale),
            )
        if self._settled_update is not None and not self._stale:
            self._write_settled_update(position, self._settled_update)
        self._db.execute("DELETE FROM pending")
        self._db.execute("DELETE FROM brush_blocks")

    def _apply_changes(self, position, *, restore_before):
        column = "before" if restore_before else "after"
        self._db.execute(f"""DELETE FROM masks WHERE (layer, cutoff, z, y, x) IN
            (SELECT layer, cutoff, z, y, x FROM changes WHERE position=? AND {column} IS NULL)""", (position,))
        self._db.execute(f"""INSERT OR REPLACE INTO masks
            SELECT layer, cutoff, z, y, x, {column} FROM changes
            WHERE position=? AND {column} IS NOT NULL""", (position,))

    def undo(self, *, expected_revision=None):
        return self._step_history(backward=True, expected_revision=expected_revision)

    def redo(self, *, expected_revision=None):
        return self._step_history(backward=False, expected_revision=expected_revision)

    def _step_history(self, *, backward, expected_revision):
        with self._database_transaction():
            position, revision = self._db.execute("SELECT position, revision FROM state WHERE id=1").fetchone()
            _check_revision(expected_revision, revision)
            action = position if backward else position + 1
            row = self._db.execute("SELECT stale_before, stale_after FROM actions WHERE position=?", (action,)).fetchone()
            if row is None:
                return False
            self._apply_changes(action, restore_before=backward)
            target, stale = (position - 1, row[0]) if backward else (position + 1, row[1])
            self._db.execute(
                "UPDATE state SET position=?, stale=?, revision=revision+1 WHERE id=1", (target, stale),
            )
            return True

    def _cutoff(self, cutoff):
        try:
            if isinstance(cutoff, (bool, np.bool_)):
                raise TypeError
            cutoff = operator.index(cutoff)
        except TypeError as error:
            raise ValueError(f"Unrecognized cutoff {cutoff}") from error
        if cutoff not in self.cutoffs:
            raise ValueError(f"Unrecognized cutoff {cutoff}")
        return cutoff

    def _block_index(self, block_zyx):
        block = _coords(block_zyx, "Block index")
        if any(index < 0 or index * size >= limit for index, size, limit in zip(block, self.block_shape, self.shape)):
            raise ValueError("Native review block lies outside the source volume")
        return block

    def _key(self, layer_id, cutoff, block_zyx, *, painted=False):
        if layer_id not in self.layers:
            raise KeyError(layer_id)
        return (layer_id, cutoff if painted else self._cutoff(cutoff), *self._block_index(block_zyx))

    def block_bounds(self, block_zyx):
        lo = tuple(index * size for index, size in zip(block_zyx, self.block_shape))
        hi = tuple(min(first + size, limit) for first, size, limit in zip(lo, self.block_shape, self.shape))
        return lo, hi

    def block_extent(self, block_zyx):
        lo, hi = self.block_bounds(block_zyx)
        return tuple(b - a for a, b in zip(lo, hi))

    def _base_block(self, key):
        layer_id, cutoff, *block = key
        mask = self.store.read_block(layer_id, cutoff, tuple(block)) if cutoff != _PAINTED else None
        if mask is None:
            return np.zeros(self.block_extent(block), dtype=bool)
        return np.asarray(mask, dtype=bool).copy()

    def _lookup(self, key, *, include_pending=True):
        packed_size = (math.prod(self.block_extent(key[2:])) + 7) // 8
        limit = packed_size + packed_size // 100 + 128
        row = None
        if self._editing and include_pending:
            row = self._db.execute(
                f"SELECT length(after), substr(after, 1, ?) FROM pending WHERE {_BLOCK_LOOKUP}", (limit, *key),
            ).fetchone()
        if row is None:
            row = self._db.execute(
                f"SELECT length(payload), substr(payload, 1, ?) FROM masks WHERE {_BLOCK_LOOKUP}", (limit, *key),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        if row[0] > limit:
            raise ValueError("Native-review journal contains an oversized block")
        return row[1]

    def _read_block(self, key, *, include_pending=True):
        payload = self._lookup(key, include_pending=include_pending)
        if payload is None:
            return self._base_block(key)
        return _unpack(payload, self.block_extent(key[2:]))

    def read_block(self, layer_id, cutoff, block_zyx):
        return self._read_block(self._key(layer_id, cutoff, block_zyx))

    def iter_override_blocks(self, layer_id, cutoff):
        """Yield committed native blocks with active overrides in ZYX order."""
        lid, cent, *_ = self._key(layer_id, cutoff, (0, 0, 0))
        yield from self._db.execute(
            "SELECT z,y,x FROM masks WHERE layer=? AND cutoff=? ORDER BY z,y,x", (lid, cent),
        )

    def iter_recent_block_deltas(self, layer_id, cutoff, previous_state):
        """Return exact before/after blocks if the display is one action behind.

        Use with `snapshot`. A skipped generation can include an undo followed
        by a new branch, so callers rebuild from committed overrides in that case.
        """
        lid, cent, *_ = self._key(layer_id, cutoff, (0, 0, 0))
        revision, position = self.display_state
        previous_revision, previous_position = previous_state
        if revision != previous_revision + 1 or abs(position - previous_position) != 1:
            return None
        action = max(position, previous_position)
        backwards = position < previous_position
        maximum = (math.prod(self.block_shape) + 7) // 8
        limit = maximum + maximum // 100 + 128

        def changes():
            for z, y, x, old_size, old, new_size, new in self._db.execute(
                """SELECT z,y,x,length(before),substr(before,1,?),length(after),substr(after,1,?)
                FROM changes WHERE position=? AND layer=? AND cutoff=? ORDER BY z,y,x""",
                (limit, limit, action, lid, cent),
            ):
                block = self._block_index((z, y, x))
                shape = self.block_extent(block)
                if (old_size is not None and len(old) != old_size
                        or new_size is not None and len(new) != new_size):
                    raise ValueError("Native-review journal contains an oversized display block")
                if old is None or new is None:
                    base = self._base_block((lid, cent, *block))
                before = base if old is None else _unpack(old, shape)
                after = base if new is None else _unpack(new, shape)
                yield (block, after, before) if backwards else (block, before, after)

        return changes()

    def read_packed_block(self, layer_id, cutoff, block_zyx):
        """Read little-endian bit-packed X rows, including staged edits."""
        key = self._key(layer_id, cutoff, block_zyx)
        return self._packed_block(key, self._lookup(key))

    def _packed_block(self, key, payload):
        if payload is None:
            return self.store.read_packed_block(key[0], key[1], key[2:])
        shape = self.block_extent(key[2:])
        if shape[2] % 8 == 0:
            return _inflate_bits(payload, math.prod(shape)).reshape(shape[0], shape[1], shape[2] // 8)
        # Journal payloads pack the whole block. A partial edge needs row padding.
        return np.packbits(_unpack(payload, shape), axis=-1, bitorder="little")

    def iter_baseline_packed_union(self, cutoff):
        """Stream the source's packed additive union when journal overlays cannot change it."""
        cutoff = self._cutoff(cutoff)
        read = getattr(self.store, "iter_packed_union", None)
        if read is None:
            return None
        changed = self._db.execute(
            "SELECT layer FROM masks WHERE cutoff=? UNION SELECT layer FROM pending WHERE cutoff=?",
            (cutoff, cutoff),
        )
        if any(layer in self._additive_ids for layer, in changed):
            return None
        return read(self._additive_ids, cutoff)

    def read_before_block(self, layer_id, cutoff, block_zyx):
        """Read the settled mask from before the current gesture's staged edits."""
        return self._read_block(self._key(layer_id, cutoff, block_zyx), include_pending=False)

    def read_painted_block(self, layer_id, block_zyx):
        return self._read_block(self._key(layer_id, _PAINTED, block_zyx, painted=True))

    def _write_block(self, key, mask):
        self._require_edit()
        actual = np.asarray(mask)
        if (actual.shape != self.block_extent(key[2:]) or actual.dtype not in (np.bool_, np.uint8)
                or (actual.dtype == np.uint8 and actual.max() > 1)):
            raise ValueError("Native edit block must match its extent and contain binary values")
        current = self._lookup(key)
        old_mask = self._base_block(key) if current is None else _unpack(current, actual.shape)
        if np.array_equal(actual, old_mask):
            return False
        # Returning to source values evicts a block from the active overrides.
        matches_source = current is not None and np.array_equal(actual, self._base_block(key))
        payload = None if matches_source else _pack(actual)
        self._stage_payload(key, payload)
        return True

    def _stage_payload(self, key, payload):
        self._db.execute("""INSERT INTO pending
            SELECT ?, ?, ?, ?, ?, (SELECT payload FROM masks
                WHERE layer=? AND cutoff=? AND z=? AND y=? AND x=?), ?
            WHERE true ON CONFLICT(layer,cutoff,z,y,x) DO UPDATE SET after=excluded.after""",
            (*key, *key, payload))

    def clip_packed_block(self, layer_id, cutoff, block_zyx, keep):
        """Intersect one contributor with a Global in row-packed native coordinates."""
        self._require_edit()
        key = self._key(layer_id, cutoff, block_zyx)
        current = self._lookup(key)
        bits = self._packed_block(key, current)
        if keep.shape != bits.shape or keep.dtype != np.uint8:
            raise ValueError("The packed Global must match the contributor block")
        retained = bits & keep
        if np.array_equal(bits, retained):
            return False
        if current is not None and np.array_equal(retained, self.store.read_packed_block(layer_id, cutoff, block_zyx)):
            payload = None
        else:
            width = self.block_extent(block_zyx)[2]
            if width % 8:
                # Archive blocks pad each X row. History packs the entire block.
                retained = np.packbits(np.unpackbits(retained, axis=-1, count=width, bitorder="little"),
                                       bitorder="little")
            payload = zlib.compress(retained, level=1)
        self._stage_payload(key, payload)
        return True

    def write_block(self, layer_id, cutoff, block_zyx, mask):
        return self._write_block(self._key(layer_id, cutoff, block_zyx), mask)

    def write_painted_block(self, layer_id, block_zyx, mask):
        return self._write_block(self._key(layer_id, _PAINTED, block_zyx, painted=True), mask)

    def _iter_journal_blocks(self, layer_id, cutoff):
        # Read each batch to completion before yielding so callers may stage
        # edits on the same connection while visiting the immutable coordinates.
        tail = "WHERE layer=? AND cutoff=? AND (z,y,x) > (?,?,?)"
        base = f"SELECT z,y,x FROM masks {tail}"
        sql = f"{base} UNION SELECT z,y,x FROM pending {tail}" if self._editing else base
        sql += " ORDER BY z,y,x LIMIT 256"
        point = (-1, -1, -1)
        while True:
            args = (layer_id, cutoff, *point) * (2 if self._editing else 1)
            batch = self._db.execute(sql, args).fetchall()
            if not batch:
                return
            yield from batch
            point = batch[-1]

    def iter_blocks(self, layer_id, cutoff):
        """Yield sorted candidate coordinates. A cleared block can remain indexed."""
        cutoff = self._key(layer_id, cutoff, (0, 0, 0))[1]
        previous = None
        for block in heapq.merge(self.store.iter_blocks(layer_id, cutoff), self._iter_journal_blocks(layer_id, cutoff)):
            if block != previous:
                yield block
                previous = block

    def iter_painted_blocks(self, layer_id):
        self._key(layer_id, _PAINTED, (0, 0, 0), painted=True)
        yield from self._iter_journal_blocks(layer_id, _PAINTED)

    def layers_at(self, block_zyx, cutoff=None):
        """List candidate contributors for a block, including staged changes."""
        block = self._block_index(block_zyx)
        if cutoff is None:
            test, args = "cutoff>=0", block
        else:
            cutoff = self._cutoff(cutoff)
            test, args = "cutoff=?", (*block, cutoff)
        clause = f"WHERE z=? AND y=? AND x=? AND {test}"
        sql = f"SELECT DISTINCT layer FROM masks {clause}"
        if self._editing:
            sql += f" UNION SELECT layer FROM pending {clause}"
            args *= 2
        present = set(self.store.layers_at(block, cutoff))
        present.update(row[0] for row in self._db.execute(sql, args))
        return tuple(layer for layer in self.layers if layer in present)

    def _region_blocks(self, lo, hi):
        axes = (range(a // s, (b - 1) // s + 1) for a, b, s in zip(lo, hi, self.block_shape))
        for block in itertools.product(*axes):
            first, stop = self.block_bounds(block)
            start = tuple(max(a, b) for a, b in zip(first, lo))
            end = tuple(min(a, b) for a, b in zip(stop, hi))
            target = tuple(slice(a - c, b - c) for a, b, c in zip(start, end, lo))
            source = tuple(slice(a - c, b - c) for a, b, c in zip(start, end, first))
            yield block, source, target

    def _region(self, start_zyx, stop_zyx):
        lo, hi = _coords(start_zyx, "Region start"), _coords(stop_zyx, "Region stop")
        if any(not 0 <= a < b <= size for a, b, size in zip(lo, hi, self.shape)):
            raise ValueError("Native review region lies outside the source volume")
        shape = tuple(b - a for a, b in zip(lo, hi))
        if math.prod(shape) > self.max_region_bytes:
            raise MemoryError(f"Native review region exceeds the {self.max_region_bytes:,}-byte decode budget")
        return lo, hi, shape

    def _region_query(self, lo, hi):
        """Translate a half-open native window to a bounded block query."""
        return _region_filter(tuple(n for a, b, step in zip(lo, hi, self.block_shape)
                                    for n in (a // step, (b - 1) // step)))

    def _region_snapshot(self):
        return nullcontext() if self._db.in_transaction else self.snapshot()

    def _region_edits(self, layer_id, cutoff, region_filter):
        max_size = (math.prod(self.block_shape) + 7) // 8
        limit = max_size + max_size // 100 + 128
        spatial_clause, bounds = region_filter
        args = limit, layer_id, cutoff, *bounds
        clause = f"WHERE layer=? AND cutoff=? AND {spatial_clause} ORDER BY z,y,x"
        masks = self._db.execute(f"SELECT z,y,x,length(payload),substr(payload, 1, ?) FROM masks {clause}", args)
        if not self._editing:
            yield from masks
            return
        pending = self._db.execute(f"SELECT z,y,x,length(after),substr(after, 1, ?) FROM pending {clause}", args)

        def coordinate(row):
            return row[:3]

        for _, candidates in itertools.groupby(heapq.merge(masks, pending, key=coordinate), key=coordinate):
            # Pending is last for a duplicate coordinate. Its NULL payload
            # removes the override and returns the immutable chunk.
            for row in candidates:
                latest = row
            yield latest

    def _sparse_region(self, layer_id, cutoff, lo, hi, shape, query):
        base = ((row[:3], False, row[3:]) for row in self.store._region_rows(layer_id, cutoff, query))
        edits = ((row[:3], True, row[3:]) for row in self._region_edits(layer_id, cutoff, query))
        result = np.zeros(shape, dtype=bool)
        for block, rows in itertools.groupby(heapq.merge(base, edits, key=operator.itemgetter(0)),
                                            key=operator.itemgetter(0)):
            stored = override = None
            for _, is_override, row in rows:
                if is_override:
                    override = row
                else:
                    stored = row
            first, end = self.block_bounds(block)
            chunk_shape = tuple(b - a for a, b in zip(first, end))
            if override is not None and override[0] is not None:
                size = (math.prod(chunk_shape) + 7) // 8
                if override[0] > size + size // 100 + 128:
                    raise ValueError("Native-review journal contains an oversized block")
                chunk = _unpack(override[1], chunk_shape)
            elif stored is not None:
                depth, height, width = chunk_shape
                packed = self.store._decode_packed_row(
                    layer_id, cutoff, block, stored, (depth, height, (width + 7) // 8), width,
                )
                chunk = np.unpackbits(packed, axis=-1, count=width, bitorder="little").view(bool)
            else:
                continue
            start = tuple(max(a, b) for a, b in zip(first, lo))
            stop = tuple(min(a, b) for a, b in zip(end, hi))
            source = tuple(slice(a - c, b - c) for a, b, c in zip(start, stop, first))
            target = tuple(slice(a - c, b - c) for a, b, c in zip(start, stop, lo))
            result[target] = chunk[source]
        return result

    def read_region(self, layer_id, cutoff, start_zyx, stop_zyx):
        """Load a bounded subvolume in native coordinates."""
        lo, hi, shape = self._region(start_zyx, stop_zyx)
        if hasattr(self.store, "_region_rows"):
            _, cutoff, *_ = self._key(layer_id, cutoff, (0, 0, 0))
            with self._region_snapshot():
                return self._sparse_region(layer_id, cutoff, lo, hi, shape, self._region_query(lo, hi))
        result = np.empty(shape, dtype=bool)
        for block, source, target in self._region_blocks(lo, hi):
            result[target] = self.read_block(layer_id, cutoff, block)[source]
        return result

    def iter_regions(self, cutoff, start_zyx, stop_zyx):
        """Yield nonempty layer windows over one revision, one layer at a time."""
        cutoff = self._cutoff(cutoff)
        lo, hi, shape = self._region(start_zyx, stop_zyx)
        with self._region_snapshot():
            if hasattr(self.store, "_region_layers"):
                region_filter = self._region_query(lo, hi)
                clause, bounds = region_filter
                args = cutoff, *bounds
                candidates = {row[0] for row in self.store._region_layers(cutoff, region_filter)}
                candidates.update(row[0] for row in self._db.execute(
                    f"SELECT DISTINCT layer FROM masks INDEXED BY mask_location "
                    f"WHERE cutoff=? AND {clause}", args,
                ))
                if self._editing:
                    candidates.update(row[0] for row in self._db.execute(
                        f"SELECT DISTINCT layer FROM pending INDEXED BY pending_location "
                        f"WHERE cutoff=? AND {clause}", args,
                    ))
                def mask_for(lid):
                    return self._sparse_region(lid, cutoff, lo, hi, shape, region_filter)
            else:
                candidates = {lid for coord, _, _ in self._region_blocks(lo, hi)
                              for lid in self.layers_at(coord, cutoff)}

                def mask_for(lid):
                    return self.read_region(lid, cutoff, lo, hi)

            for lid in self.layers:
                if lid in candidates:
                    mask = mask_for(lid)
                    if mask.any():
                        yield lid, mask

    def apply_region_diff(self, layer_id, cutoff, start_zyx, before, after):
        """Sync only pixels edited in the loaded region, with normal cutoff scope.

        Erase clears all cutoff memberships. Paint adds at the displayed and
        looser cutoffs and retains the painted seed for later marker operations.
        """
        self._require_edit()
        old, new = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
        if old.ndim != 3 or old.shape != new.shape:
            raise ValueError("Brush changes require equal ZYX arrays")
        start = _coords(start_zyx, "Region start")
        stop = tuple(a + n for a, n in zip(start, old.shape))
        lo, hi, _ = self._region(start, stop)
        cutoff = self._key(layer_id, cutoff, (0, 0, 0))[1]
        position = self.cutoffs.index(cutoff)
        changed = 0
        for block, source, target in self._region_blocks(lo, hi):
            old_window, new_window = old[target], new[target]
            erase, paint = old_window & ~new_window, new_window & ~old_window
            difference = erase | paint
            if not difference.any():
                continue
            displayed = self.read_block(layer_id, cutoff, block)
            if np.any((displayed[source] != old_window) & difference):
                raise RuntimeError("The displayed native region changed before the brush edit could be saved")
            for index, value in enumerate(self.cutoffs):
                mask = displayed if value == cutoff else self.read_block(layer_id, value, block)
                mask[source] &= ~erase
                if index <= position:
                    mask[source] |= paint
                self.write_block(layer_id, value, block, mask)
            painted = self.read_painted_block(layer_id, block)
            painted[source] &= ~erase
            painted[source] |= paint
            self.write_painted_block(layer_id, block, painted)
            if layer_id in self._additive_ids and erase.any():
                self._db.execute("INSERT OR IGNORE INTO brush_blocks VALUES (?, ?, ?)", block)
            changed += int(np.count_nonzero(difference))
        return changed

    def _rank(self, block, *, include_pending):
        rank = np.zeros(self.block_extent(block), dtype=np.int16)
        for ordinal, cutoff in enumerate(self.cutoffs, 1):
            for lid in self.layers_at(block, cutoff):
                if lid not in self._additive_ids:
                    continue
                rank[self._read_block((lid, cutoff, *block), include_pending=include_pending)] = ordinal
        return rank

    def _follow_brush_global(self):
        if self.global_layer_id is None:
            return
        blocks = self._db.execute("SELECT z,y,x FROM brush_blocks ORDER BY z,y,x")
        for block in blocks:
            old = self._rank(block, include_pending=False)
            new = self._rank(block, include_pending=True)
            fell = new < old
            if not fell.any():
                continue
            self._stale = True
            for ordinal, cutoff in enumerate(self.cutoffs, 1):
                mask = self.read_block(self.global_layer_id, cutoff, block)
                mask &= ~(fell & (new < ordinal))
                self.write_block(self.global_layer_id, cutoff, block, mask)

    def _global_slab_reader(self, cutoff):
        # Read only indexed XY blocks in the current Z band. Avoid one lookup
        # per absent tile during a whole native export.
        lid = self.global_layer_id
        bands = itertools.groupby(self.iter_blocks(lid, cutoff), key=operator.itemgetter(0))
        current = next(bands, None)
        band_z, keys = -1, ()
        block_depth = self.block_shape[0]

        def read(z0, z1):
            nonlocal band_z, keys, current
            output = np.zeros((z1 - z0, *self.shape[1:]), dtype=bool)
            for bz in range(z0 // block_depth, (z1 - 1) // block_depth + 1):
                if band_z != bz:
                    while current is not None and current[0] < bz:
                        current = next(bands, None)
                    if current is not None and current[0] == bz:
                        keys = tuple(current[1])
                        current = next(bands, None)
                    else:
                        keys = ()
                    band_z = bz
                for block in keys:
                    (start, y, x), (stop, y1, x1) = self.block_bounds(block)
                    first, last = max(start, z0), min(stop, z1)
                    mask = self.read_block(lid, cutoff, block)
                    output[first - z0:last - z0, y:y1, x:x1] = mask[first - start:last - start]
            return output

        return read

    def export_global(self, output, cutoff, *, memory_mib=256, expected_revision=None):
        """Write one native binary NRRD from a consistent view of the edit log."""
        from .json_publication import _fsync_parent_directory
        from .reconciliation_io import ReferenceGeometry, write_seg_nrrd
        from .run_transport import _publish_fresh

        lid = self.global_layer_id
        if lid is None:
            raise ValueError("The native review has no Global layer to export")
        self._key(lid, cutoff, (0, 0, 0))
        output = Path(output).expanduser().absolute()
        if os.path.lexists(output):
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        geometry = ReferenceGeometry(
            self.shape, space=self.store.geometry["space"],
            directions_xyz=self.store.geometry["directions_xyz"], origin_xyz=self.store.geometry["origin_xyz"],
        )
        with tempfile.TemporaryDirectory(prefix=".native-review-export-", dir=output.parent) as staging:
            temporary = Path(staging) / "cleaned.nrrd"
            with self.snapshot() as revision:
                _check_revision(expected_revision, revision)
                write_seg_nrrd(
                    temporary, shape_tyx=self.shape, geometry=geometry, read_slab=self._global_slab_reader(cutoff),
                    segment_name="Global", memory_mib=min(memory_mib, self.max_region_bytes * 3 / _MIB),
                    chunk_slices=self.block_shape[0],
                )
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            _publish_fresh(temporary, output)
            _fsync_parent_directory(output.parent)
        return output
