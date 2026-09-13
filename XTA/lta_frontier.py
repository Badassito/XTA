"""Disk-backed full-seed mailboxes for canonical, bounded LTA relay waves.

Keys retain exact lineage, tile, prompt frame and direction, but never a route
or generation. Canonical 29-frame advances join downstream arrivals at shared
boundaries. Distinct mailboxes are finite; inference attempts are not limited
to one per mailbox, because genuinely new pixels may require another attempt.
No pixel-difference prompt or novelty threshold is used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np

from .lta_coverage import _binary, _crop, _decode, _union_crops
from .lta_propagation import LtaMaskSeed, LtaSeedProvenance
from .lta_tile_tracking import LtaLineageId
from .lta_windows import LTA_SESSION_FRAMES, WindowPlan


FRONTIER_SCHEMA = "lta.canonical-frontier/1"
_ADVANCE = LTA_SESSION_FRAMES - 1
_DIRECTIONS = {"forward": 0, "backward": 1}
_PROVENANCES = tuple(value.value for value in LtaSeedProvenance)


def _index(value, name, *, minimum=0):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(operator.index(value))
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _revision(shape, crop):
    digest = hashlib.sha256()
    digest.update(_json({"shape": list(shape), "crop": list(crop[:4]),
                         "encoding": "cropped-flat-packbits-little"}).encode("utf-8"))
    digest.update(crop[4])
    return digest.hexdigest()


def _canonical_window(frame, direction, frame_count):
    if direction == "forward":
        start = frame
        stop = min((frame // _ADVANCE + 1) * _ADVANCE, frame_count - 1) + 1
        # The terminal mailbox follows the final advancing slab.
        sweep = ((frame_count - 1 + _ADVANCE - 1) // _ADVANCE
                 if frame == frame_count - 1 else frame // _ADVANCE)
    else:
        start = 0 if frame == 0 else ((frame - 1) // _ADVANCE) * _ADVANCE
        stop = frame + 1
        sweep = 1 if frame == 0 else -(start // _ADVANCE)
    return WindowPlan(branch=direction, ordinal=0, frame_start=start,
                      frame_stop=stop, prompt_frame=frame, direction=direction,
                      seed_kind="spatial_relay"), sweep


@dataclass(frozen=True)
class FrontierEntry:
    """One immutable, leased full-input revision; acknowledge this exact value."""

    key: str
    seed: LtaMaskSeed
    tile_index: int
    direction: str
    window: WindowPlan
    revision_sha256: str


class CanonicalFrontier:
    """One coordinator's SQLite frontier, with at most one outstanding wave.

    ``offer`` returns whether foreground was added, including to an already
    dirty mailbox. Offers during a lease cannot change its frozen seed;
    ``complete`` leaves any subsequently added bits dirty. Taking another wave
    before completing the outstanding lease is rejected. Closing and reopening
    retains dirty inputs, but process-local attempt counters restart.
    """

    def __init__(self, path, frame_count):
        self.path = Path(path).resolve(strict=False)
        self.frame_count = _index(frame_count, "frame_count", minimum=1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._lease: tuple[FrontierEntry, ...] = ()
        self._audit = Counter()
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA cache_size=-4096")
            self._db.execute("PRAGMA temp_store=FILE")
            tables = {row[0] for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and "frontier_meta" not in tables:
                raise ValueError("frontier path contains an unrelated SQLite database")
            if not tables:
                self._db.executescript("""
                    CREATE TABLE frontier_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE lineages (token TEXT PRIMARY KEY, metadata TEXT NOT NULL);
                    CREATE TABLE tiles (scope TEXT PRIMARY KEY, height INTEGER NOT NULL, width INTEGER NOT NULL);
                    CREATE TABLE mailboxes (
                        key TEXT PRIMARY KEY, identity TEXT NOT NULL, lineage_token TEXT NOT NULL,
                        tile_scope TEXT NOT NULL, tile_index INTEGER NOT NULL, frame INTEGER NOT NULL,
                        direction TEXT NOT NULL, direction_order INTEGER NOT NULL, sweep INTEGER NOT NULL,
                        y INTEGER NOT NULL, x INTEGER NOT NULL, height INTEGER NOT NULL, width INTEGER NOT NULL,
                        packed BLOB NOT NULL, foreground INTEGER NOT NULL, probability REAL NOT NULL,
                        visited TEXT NOT NULL, provenances INTEGER NOT NULL, generation INTEGER NOT NULL,
                        revision TEXT NOT NULL, processed_revision TEXT, dirty INTEGER NOT NULL
                    );
                    CREATE INDEX ready_wave ON mailboxes(dirty,direction_order,sweep,key);
                """)
                self._db.executemany("INSERT INTO frontier_meta VALUES (?,?)",
                                     (("schema", FRONTIER_SCHEMA), ("frame_count", str(self.frame_count))))
                self._db.commit()
            metadata = dict(self._db.execute("SELECT key,value FROM frontier_meta"))
            if metadata != {"schema": FRONTIER_SCHEMA, "frame_count": str(self.frame_count)}:
                raise ValueError("frontier database schema/frame count differs from this run")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            count, dirty, stored, foreground = self._db.execute(
                "SELECT COUNT(*),COALESCE(SUM(dirty),0),COALESCE(SUM(length(packed)),0),COALESCE(SUM(foreground),0) FROM mailboxes"
            ).fetchone()
            self._totals = dict(mailbox_count=count, dirty_count=dirty,
                                stored_mask_bytes=stored, foreground_pixels=foreground)
        except BaseException:
            self._db.close()
            self._closed = True
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("canonical frontier is closed")

    def offer(self, seed, *, tile_index, direction):
        """Merge a seed at its actual frame; never move pixels between frames."""
        self._ensure_open()
        if not isinstance(seed, LtaMaskSeed) or not isinstance(seed.lineage, LtaLineageId):
            raise TypeError("frontier offers require an LtaMaskSeed with an LtaLineageId")
        if direction not in _DIRECTIONS:
            raise ValueError("frontier direction must be forward or backward")
        tile = _index(tile_index, "tile_index")
        frame = _index(seed.frame_index, "seed frame_index")
        if frame >= self.frame_count:
            raise ValueError("frontier seed frame is outside the run")
        _index(seed.object_id, "seed object_id")
        generation = _index(seed.relay_generation, "seed relay_generation")
        provenance = LtaSeedProvenance.coerce(seed.provenance)
        provenance_bit = 1 << _PROVENANCES.index(provenance.value)
        probability = float(seed.tracker_probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("frontier probability must be finite and in [0,1]")
        visited = sorted({_index(value, "visited tile index") for value in seed.visited_tile_indices})
        mask = _binary(seed.mask)
        crop = _crop(mask)
        if crop is None:
            raise ValueError("frontier seeds must contain foreground")
        shape = tuple(mask.shape)
        lineage_record = _json(asdict(seed.lineage))
        token = seed.lineage.token
        scope = _json([seed.lineage.volume_id, seed.lineage.physical_view_id,
                       seed.lineage.runtime_view_id, seed.lineage.tile_config_id, tile])
        identity = _json({"lineage": asdict(seed.lineage), "tile_index": tile,
                          "frame_index": frame, "direction": direction})
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        _window, sweep = _canonical_window(frame, direction, self.frame_count)
        with self._db:
            known_lineage = self._db.execute("SELECT metadata FROM lineages WHERE token=?", (token,)).fetchone()
            if known_lineage is not None and known_lineage[0] != lineage_record:
                raise ValueError("frontier lineage token collides with different metadata")
            known_shape = self._db.execute("SELECT height,width FROM tiles WHERE scope=?", (scope,)).fetchone()
            if known_shape is not None and tuple(known_shape) != shape:
                raise ValueError("frontier seed shape differs from its tile geometry")
            previous = self._db.execute("SELECT * FROM mailboxes WHERE key=?", (key,)).fetchone()
            if previous is not None and previous["identity"] != identity:
                raise ValueError("frontier key collides with different metadata")
            if previous is not None:
                old_crop = tuple(previous[name] for name in ("y", "x", "height", "width", "packed", "foreground"))
                if _revision(shape, old_crop) != previous["revision"]:
                    raise RuntimeError("frontier stored mask revision changed")
                crop = _union_crops(old_crop, crop)
                novel = crop[5] > old_crop[5]
                probability = max(probability, previous["probability"])
                visited = sorted(set(visited) | set(json.loads(previous["visited"])))
                provenance_bit |= previous["provenances"]
                generation = max(generation, previous["generation"])
            else:
                novel = True
            revision = _revision(shape, crop)
            dirty = int(novel or (previous is not None and previous["dirty"]))
            self._db.execute("INSERT OR IGNORE INTO lineages VALUES (?,?)", (token, lineage_record))
            self._db.execute("INSERT OR IGNORE INTO tiles VALUES (?,?,?)", (scope, *shape))
            self._db.execute("""
                INSERT INTO mailboxes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET y=excluded.y,x=excluded.x,height=excluded.height,
                    width=excluded.width,packed=excluded.packed,foreground=excluded.foreground,
                    probability=excluded.probability,visited=excluded.visited,provenances=excluded.provenances,
                    generation=excluded.generation,revision=excluded.revision,dirty=excluded.dirty
            """, (key, identity, token, scope, tile, frame, direction, _DIRECTIONS[direction], sweep,
                  *crop, probability, _json(visited), provenance_bit, generation, revision,
                  None if previous is None else previous["processed_revision"], dirty))
        self._audit["offer_count"] += 1
        self._audit["novel_offer_count" if novel else "metadata_only_offer_count"] += 1
        self._totals["mailbox_count"] += previous is None
        self._totals["dirty_count"] += dirty - (0 if previous is None else previous["dirty"])
        self._totals["stored_mask_bytes"] += len(crop[4]) - (0 if previous is None else len(previous["packed"]))
        self._totals["foreground_pixels"] += crop[5] - (0 if previous is None else previous["foreground"])
        return bool(novel)

    def take_wave(self, limit=32):
        """Lease only the first direction/slab group, bounded independently of GPUs."""
        self._ensure_open()
        limit = _index(limit, "wave limit", minimum=1)
        if self._lease:
            raise RuntimeError("complete the outstanding frontier wave before taking another")
        group = self._db.execute(
            "SELECT direction_order,sweep FROM mailboxes WHERE dirty=1 ORDER BY direction_order,sweep,key LIMIT 1"
        ).fetchone()
        if group is None:
            return ()
        rows = self._db.execute("""
            SELECT m.*,l.metadata,t.height AS tile_height,t.width AS tile_width
            FROM mailboxes m JOIN lineages l ON l.token=m.lineage_token JOIN tiles t ON t.scope=m.tile_scope
            WHERE dirty=1 AND direction_order=? AND sweep=? ORDER BY key LIMIT ?
        """, (*group, limit)).fetchall()
        entries = []
        for row in rows:
            shape = row["tile_height"], row["tile_width"]
            crop = tuple(row[name] for name in ("y", "x", "height", "width", "packed", "foreground"))
            if (_revision(shape, crop) != row["revision"] or min(crop[:2]) < 0
                    or min(crop[2:4]) < 1 or crop[0] + crop[2] > shape[0] or crop[1] + crop[3] > shape[1]
                    or len(crop[4]) != (crop[2] * crop[3] + 7) // 8):
                raise RuntimeError("frontier stored mask geometry/revision changed")
            decoded = _decode(crop)
            if int(np.count_nonzero(decoded)) != crop[5]:
                raise RuntimeError("frontier stored foreground count changed")
            mask = np.zeros(shape, dtype=np.bool_)
            mask[crop[0]:crop[0] + crop[2], crop[1]:crop[1] + crop[3]] = decoded
            lineage = LtaLineageId(**json.loads(row["metadata"]))
            identity = _json({"lineage": asdict(lineage), "tile_index": row["tile_index"],
                              "frame_index": row["frame"], "direction": row["direction"]})
            if identity != row["identity"] or hashlib.sha256(identity.encode("utf-8")).hexdigest() != row["key"]:
                raise RuntimeError("frontier stored identity changed")
            seed = LtaMaskSeed(
                lineage=lineage, frame_index=row["frame"], object_id=0, mask=mask,
                provenance=LtaSeedProvenance.SPATIAL_RELAY, tracker_probability=row["probability"],
                relay_generation=row["generation"], visited_tile_indices=tuple(json.loads(row["visited"])),
                source_receipt={"schema": FRONTIER_SCHEMA, "frontier_key": row["key"],
                                "frontier_revision_sha256": row["revision"],
                                "input_provenances": [value for bit, value in enumerate(_PROVENANCES)
                                                      if row["provenances"] & (1 << bit)]},
            )
            window, sweep = _canonical_window(seed.frame_index, row["direction"], self.frame_count)
            if sweep != row["sweep"] or _DIRECTIONS[row["direction"]] != row["direction_order"]:
                raise RuntimeError("frontier stored sweep geometry changed")
            entries.append(FrontierEntry(row["key"], seed, row["tile_index"], row["direction"], window, row["revision"]))
        self._lease = tuple(entries)
        self._audit["wave_count"] += 1
        return self._lease

    def complete(self, entries: Iterable[FrontierEntry]):
        """Acknowledge exactly the outstanding frozen input revisions, atomically."""
        self._ensure_open()
        values = tuple(entries)
        if not self._lease or len(values) != len(self._lease) or any(
                supplied is not leased for supplied, leased in zip(values, self._lease)):
            raise ValueError("frontier completion does not match the outstanding lease")
        dirty_delta = 0
        with self._db:
            for entry in values:
                row = self._db.execute("SELECT revision,dirty FROM mailboxes WHERE key=?", (entry.key,)).fetchone()
                if row is None:
                    raise RuntimeError("leased frontier mailbox disappeared")
                if _revision(tuple(entry.seed.mask.shape), _crop(entry.seed.mask)) != entry.revision_sha256:
                    raise ValueError("leased frontier seed revision changed")
                dirty = int(row["revision"] != entry.revision_sha256)
                dirty_delta += dirty - row["dirty"]
                self._db.execute("UPDATE mailboxes SET processed_revision=?,dirty=? WHERE key=?",
                                 (entry.revision_sha256, dirty, entry.key))
        self._totals["dirty_count"] += dirty_delta
        self._audit["completed_entry_count"] += len(values)
        self._lease = ()

    def stats(self):
        self._ensure_open()
        return {"schema": FRONTIER_SCHEMA, **self._totals,
                **{key: self._audit[key] for key in ("offer_count", "novel_offer_count", "metadata_only_offer_count",
                                                    "wave_count", "completed_entry_count")},
                "leased_count": len(self._lease), "frame_count": self.frame_count,
                "database_bytes": self.path.stat().st_size,
                "wal_bytes": Path(str(self.path) + "-wal").stat().st_size if Path(str(self.path) + "-wal").exists() else 0,
                "page_cache_limit_bytes": 4 * 1024 * 1024,
                "attempt_bound": "new foreground can revisit existing mailboxes; no per-mailbox attempt cap"}

    def close(self):
        if not self._closed:
            self._db.close()
            self._lease = ()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_args):
        self.close()


__all__ = ("FRONTIER_SCHEMA", "CanonicalFrontier", "FrontierEntry")
