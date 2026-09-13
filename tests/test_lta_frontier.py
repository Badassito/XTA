from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from XTA.lta_frontier import CanonicalFrontier
from XTA.lta_propagation import LtaMaskSeed, LtaSeedProvenance
from XTA.lta_tile_tracking import LtaLineageId


class CanonicalFrontierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lineage = LtaLineageId("volume", "transverse", "tta_a0", "object", "s4_st2")

    def frontier(self, name="frontier", frames=90):
        value = CanonicalFrontier(self.root / (name + ".sqlite3"), frame_count=frames)
        self.addCleanup(value.close)
        return value

    def seed(self, *, frame=12, pixels=((1, 1),), shape=(4, 4), lineage=None,
             probability=.7, visited=(0,), generation=1, object_id=7,
             provenance=LtaSeedProvenance.SPATIAL_RELAY):
        mask = np.zeros(shape, dtype=bool)
        for point in pixels:
            mask[point] = True
        return LtaMaskSeed(
            lineage=self.lineage if lineage is None else lineage,
            frame_index=frame, object_id=object_id, mask=mask,
            provenance=provenance, tracker_probability=probability,
            visited_tile_indices=visited, relay_generation=generation,
            source_receipt={"route": "not part of canonical identity"},
        )

    def test_offer_order_does_not_change_merged_full_seed_or_revision(self):
        seeds = (
            self.seed(pixels=((0, 0),), probability=.2, visited=(0,), object_id=9),
            self.seed(pixels=((1, 1),), probability=.9, visited=(2,), generation=3,
                      provenance=LtaSeedProvenance.TEMPORAL_DOGFOOD),
            self.seed(pixels=((3, 3),), probability=.4, visited=(1,), object_id=5),
        )
        signatures = []
        for index, order in enumerate(itertools.permutations(seeds)):
            frontier = self.frontier(str(index))
            for seed in order:
                self.assertTrue(frontier.offer(seed, tile_index=1, direction="forward"))
            entries = frontier.take_wave()
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.seed.object_id, 0)
            self.assertEqual(entry.seed.tracker_probability, .9)
            self.assertEqual(entry.seed.visited_tile_indices, (0, 1, 2))
            self.assertEqual(entry.seed.relay_generation, 3)
            self.assertEqual(int(entry.seed.mask.sum()), 3)
            self.assertEqual(entry.seed.frame_index, 12)
            self.assertNotIn("route", entry.seed.source_receipt)
            signatures.append((entry.key, entry.revision_sha256, entry.seed.mask.tobytes(),
                               json.dumps(entry.seed.source_receipt, sort_keys=True)))
            frontier.complete(entries)
            self.assertEqual(frontier.take_wave(), ())
        self.assertEqual(len(set(signatures)), 1)

    def test_subsets_and_score_or_provenance_changes_do_not_redirty(self):
        frontier = self.frontier()
        first = self.seed(pixels=((0, 0), (2, 2)), probability=.3)
        frontier.offer(first, tile_index=0, direction="forward")
        frontier.complete(frontier.take_wave())
        metadata = self.seed(pixels=((0, 0),), probability=.99, visited=(2, 3),
                             generation=8, provenance=LtaSeedProvenance.TEMPORAL_DOGFOOD)
        self.assertFalse(frontier.offer(metadata, tile_index=0, direction="forward"))
        self.assertEqual(frontier.take_wave(), ())
        self.assertTrue(frontier.offer(self.seed(pixels=((1, 1),)), tile_index=0, direction="forward"))
        entry, = frontier.take_wave()
        self.assertEqual(int(entry.seed.mask.sum()), 3)
        self.assertEqual(entry.seed.tracker_probability, .99)
        self.assertEqual(entry.seed.visited_tile_indices, (0, 2, 3))
        self.assertEqual(entry.seed.relay_generation, 8)

    def test_lineage_tile_direction_and_prompt_are_independent(self):
        frontier = self.frontier()
        offers = (
            (self.seed(), 0, "forward"),
            (self.seed(lineage=replace(self.lineage, lineage_id="unmatched")), 0, "forward"),
            (self.seed(), 1, "forward"),
            (self.seed(), 0, "backward"),
            (self.seed(frame=13), 0, "forward"),
        )
        for seed, tile, direction in offers:
            frontier.offer(seed, tile_index=tile, direction=direction)
        actual = []
        while entries := frontier.take_wave():
            actual.extend((entry.seed.lineage, entry.tile_index, entry.direction, entry.seed.frame_index) for entry in entries)
            frontier.complete(entries)
        self.assertCountEqual(actual, [(seed.lineage, tile, direction, seed.frame_index) for seed, tile, direction in offers])
        self.assertEqual(frontier.stats()["mailbox_count"], 5)

    def test_every_boundary_uses_at_most_thirty_frames_without_moving_prompt(self):
        for count in (1, 29, 30, 31, 59, 90, 1929):
            frontier = self.frontier(str(count), frames=count)
            frames = {value for value in (0, 1, 14, 28, 29, 30, 57, 58, 59, 60, 87, 88, 89, count-2, count-1) if 0 <= value < count}
            for frame in frames:
                for direction in ("forward", "backward"):
                    frontier.offer(self.seed(frame=frame), tile_index=0, direction=direction)
            found = set()
            while entries := frontier.take_wave():
                for entry in entries:
                    frame = entry.seed.frame_index
                    window = entry.window
                    found.add((frame, entry.direction))
                    self.assertEqual(window.prompt_frame, frame)
                    self.assertEqual(window.seed_kind, "spatial_relay")
                    self.assertEqual(window.ordinal, 0)
                    self.assertLessEqual(window.frame_count, 30)
                    if entry.direction == "forward":
                        expected = frame, min((frame // 29 + 1) * 29, count - 1) + 1
                    else:
                        expected = (0 if frame == 0 else ((frame - 1) // 29) * 29), frame + 1
                    self.assertEqual((window.frame_start, window.frame_stop), expected)
                frontier.complete(entries)
            self.assertEqual(found, set(itertools.product(frames, ("forward", "backward"))))

    def test_wave_never_mixes_direction_or_slab_and_is_bounded(self):
        frontier = self.frontier()
        for frame in (1, 2, 28, 29, 30, 58, 59, 89):
            for direction in ("forward", "backward"):
                frontier.offer(self.seed(frame=frame), tile_index=0, direction=direction)
        groups = []
        while entries := frontier.take_wave(limit=2):
            self.assertLessEqual(len(entries), 2)
            directions = {entry.direction for entry in entries}
            self.assertEqual(len(directions), 1)
            direction = entries[0].direction
            if direction == "forward":
                slab = lambda entry: 4 if entry.seed.frame_index == 89 else entry.seed.frame_index // 29
                key = 0, slab(entries[0])
            else:
                slab = lambda entry: -(entry.window.frame_start // 29)
                key = 1, slab(entries[0])
            self.assertEqual(len({slab(entry) for entry in entries}), 1)
            self.assertEqual([entry.key for entry in entries], sorted(entry.key for entry in entries))
            groups.append(key)
            frontier.complete(entries)
        self.assertEqual(groups, sorted(groups))

    def test_many_offsets_converge_to_one_shared_outbound_revision_per_boundary(self):
        frontier = self.frontier()
        for frame in range(1, 29):
            point = (frame % 4, (frame // 4) % 4)
            frontier.offer(self.seed(frame=frame, pixels=(point,)), tile_index=1, direction="forward")
        counts = Counter()
        local_completed = 0
        while entries := frontier.take_wave(limit=7):
            frontier.complete(entries)
            for entry in entries:
                frame = entry.seed.frame_index
                counts[frame] += 1
                if frame < 29:
                    local_completed += 1
                else:
                    self.assertEqual(local_completed, 28)
                    self.assertEqual(int(entry.seed.mask.sum()), 16)
                boundary = entry.window.frame_stop - 1
                if boundary != frame:
                    frontier.offer(replace(entry.seed, frame_index=boundary,
                                           provenance=LtaSeedProvenance.TEMPORAL_DOGFOOD),
                                   tile_index=entry.tile_index, direction=entry.direction)
        self.assertEqual({frame: count for frame, count in counts.items() if frame >= 29},
                         {29: 1, 58: 1, 87: 1, 89: 1})
        self.assertEqual(frontier.stats()["completed_entry_count"], 32)
        self.assertEqual(frontier.stats()["mailbox_count"], 32)
        self.assertEqual(frontier.stats()["dirty_count"], 0)

    def test_revisions_can_revisit_one_mailbox_without_adding_nodes(self):
        frontier = self.frontier()
        keys = set()
        for index in range(16):
            frontier.offer(self.seed(pixels=((index // 4, index % 4),)), tile_index=0, direction="forward")
            entries = frontier.take_wave()
            keys.add(entries[0].key)
            self.assertEqual(int(entries[0].seed.mask.sum()), index + 1)
            frontier.complete(entries)
        self.assertEqual(len(keys), 1)
        self.assertEqual(frontier.stats()["mailbox_count"], 1)
        self.assertEqual(frontier.stats()["completed_entry_count"], 16)

    def test_frozen_lease_does_not_acknowledge_bits_offered_during_execution(self):
        frontier = self.frontier()
        frontier.offer(self.seed(pixels=((0, 0),)), tile_index=0, direction="forward")
        first = frontier.take_wave()
        frontier.offer(self.seed(pixels=((3, 3),)), tile_index=0, direction="forward")
        self.assertEqual(int(first[0].seed.mask.sum()), 1)
        frontier.complete(first)
        second = frontier.take_wave()
        self.assertEqual(first[0].key, second[0].key)
        self.assertNotEqual(first[0].revision_sha256, second[0].revision_sha256)
        self.assertEqual(int(second[0].seed.mask.sum()), 2)
        frontier.complete(second)
        self.assertEqual(frontier.take_wave(), ())

    def test_bad_lease_or_revision_cannot_partially_complete_wave(self):
        frontier = self.frontier()
        for tile in (0, 1):
            frontier.offer(self.seed(), tile_index=tile, direction="forward")
        entries = frontier.take_wave()
        with self.assertRaisesRegex(RuntimeError, "outstanding"):
            frontier.take_wave()
        for bad in (entries[:1], entries[::-1], (replace(entries[0], revision_sha256="0"*64), entries[1])):
            with self.assertRaisesRegex(ValueError, "lease"):
                frontier.complete(bad)
            self.assertEqual(frontier.stats()["dirty_count"], 2)
            self.assertEqual(frontier.stats()["completed_entry_count"], 0)
        frontier.complete(entries)
        with self.assertRaisesRegex(ValueError, "lease"):
            frontier.complete(entries)

    def test_invalid_metadata_geometry_and_lineage_collision_reject_atomically(self):
        frontier = self.frontier()
        frontier.offer(self.seed(), tile_index=0, direction="forward")
        before = frontier.stats().copy()
        for seed, tile, direction in (
            (self.seed(frame=90), 0, "forward"),
            (self.seed(shape=(5, 5)), 0, "forward"),
            (self.seed(), -1, "forward"),
            (self.seed(), 0, "both"),
        ):
            with self.assertRaises((ValueError, TypeError)):
                frontier.offer(seed, tile_index=tile, direction=direction)
        invalid = self.seed()
        object.__setattr__(invalid, "provenance", "invented")
        with self.assertRaises(ValueError):
            frontier.offer(invalid, tile_index=0, direction="forward")
        first = replace(self.lineage, volume_id="v::x", physical_view_id="p")
        second = replace(self.lineage, volume_id="v", physical_view_id="x::p")
        self.assertEqual(first.token, second.token)
        frontier.offer(self.seed(lineage=first), tile_index=0, direction="forward")
        with self.assertRaisesRegex(ValueError, "collides"):
            frontier.offer(self.seed(lineage=second), tile_index=0, direction="forward")
        self.assertEqual(frontier.stats()["mailbox_count"], before["mailbox_count"] + 1)
        self.assertEqual(frontier.stats()["offer_count"], before["offer_count"] + 1)

    def test_sqlite_storage_is_cropped_packed_and_uncompleted_inputs_survive_reopen(self):
        path = self.root / "packed.sqlite3"
        with CanonicalFrontier(path, 90) as frontier:
            frontier.offer(self.seed(shape=(1008, 1008), pixels=((900, 950),)), tile_index=0, direction="forward")
            entry, = frontier.take_wave()
            self.assertEqual(frontier.stats()["stored_mask_bytes"], 1)
            key = entry.key
        with closing(sqlite3.connect(path)) as db:
            row = db.execute("SELECT y,x,height,width,length(packed) FROM mailboxes").fetchone()
            self.assertEqual(row, (900, 950, 1, 1, 1))
        with CanonicalFrontier(path, 90) as frontier:
            entries = frontier.take_wave()
            self.assertEqual(entries[0].key, key)
            self.assertTrue(entries[0].seed.mask[900, 950])
            self.assertEqual(int(entries[0].seed.mask.sum()), 1)
            frontier.complete(entries)
        with CanonicalFrontier(path, 90) as frontier:
            self.assertEqual(frontier.take_wave(), ())
        with self.assertRaisesRegex(ValueError, "frame count"):
            CanonicalFrontier(path, 91)

    def test_invalid_limits_and_closed_operations_are_rejected(self):
        for count in (0, -1, True, 1.5):
            with self.assertRaises((ValueError, TypeError)):
                CanonicalFrontier(self.root / "invalid.sqlite3", count)
        frontier = self.frontier()
        for limit in (0, -1, True, 1.5):
            with self.assertRaises((ValueError, TypeError)):
                frontier.take_wave(limit)
        frontier.close()
        for operation in (frontier.stats, frontier.take_wave, lambda: frontier.offer(self.seed(), tile_index=0, direction="forward")):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()


if __name__ == "__main__":
    unittest.main()
