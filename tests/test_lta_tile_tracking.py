from __future__ import annotations

import unittest

import numpy as np

from XTA.lta_tile_tracking import LtaLineageId, rebase_tile_mask
from XTA.lta_tiles import eight_neighbor_graph, plan_tile_grid


def _mask(*points: tuple[int, int], size: int = 6) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.bool_)
    for row, column in points:
        mask[row, column] = True
    return mask


class LtaTileTrackingTests(unittest.TestCase):
    def test_lineage_id_token_and_rejects_empty_fields(self) -> None:
        lineage = LtaLineageId(
            volume_id=" volume ",
            physical_view_id="transverse",
            runtime_view_id="transverse::angle-0",
            lineage_id="object-7",
        )
        self.assertEqual(lineage.volume_id, "volume")
        self.assertEqual(
            lineage.token,
            "volume::transverse::transverse::angle-0::fullframe::object-7",
        )
        with self.assertRaisesRegex(ValueError, "lineage_id must not be empty"):
            LtaLineageId("volume", "transverse", "transverse::angle-0", "  ")

    def test_eight_neighbor_graph_and_cardinal_diagonal_rebasing(self) -> None:
        tiles = plan_tile_grid(
            source_width=14,
            source_height=14,
            tile_size=6,
            tile_stride=4,
        )
        graph = eight_neighbor_graph(tiles)
        center_index = 4
        center = tiles[center_index]
        edges = graph[center_index]

        self.assertEqual(len(edges), 8)
        self.assertEqual(
            {edge.direction for edge in edges},
            {
                "northwest",
                "north",
                "northeast",
                "west",
                "east",
                "southwest",
                "south",
                "southeast",
            },
        )
        for edge in edges:
            reverse = next(
                candidate
                for candidate in graph[edge.destination_index]
                if candidate.destination_index == center_index
            )
            self.assertEqual(reverse.overlap_xyxy, edge.overlap_xyxy)
            self.assertEqual(reverse.row_delta, -edge.row_delta)
            self.assertEqual(reverse.column_delta, -edge.column_delta)

            global_x, global_y = edge.overlap_xyxy[:2]
            source_mask = _mask(
                (global_y - center.top, global_x - center.left),
                size=center.size,
            )
            destination = tiles[edge.destination_index]
            translated = rebase_tile_mask(source_mask, center, destination)
            self.assertEqual(int(np.count_nonzero(translated)), 1, edge.direction)
            self.assertTrue(
                translated[
                    global_y - destination.top,
                    global_x - destination.left,
                ],
                edge.direction,
            )

    def test_edge_pinned_grid_uses_actual_overlap_rectangles(self) -> None:
        tiles = plan_tile_grid(
            source_width=13,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        graph = eight_neighbor_graph(tiles)

        self.assertEqual([tile.left for tile in tiles], [0, 4, 7])
        self.assertEqual(
            [(edge.destination_index, edge.overlap_xyxy) for edge in graph[0]],
            [(1, (4, 0, 6, 6))],
        )
        self.assertEqual(
            [(edge.destination_index, edge.overlap_xyxy) for edge in graph[1]],
            [(0, (4, 0, 6, 6)), (2, (7, 0, 10, 6))],
        )
        self.assertEqual(
            [(edge.destination_index, edge.overlap_xyxy) for edge in graph[2]],
            [(1, (7, 0, 10, 6))],
        )

    def test_touching_tiles_rebase_across_cardinal_and_diagonal_ports(self) -> None:
        tiles = plan_tile_grid(
            source_width=12,
            source_height=12,
            tile_size=6,
            tile_stride=6,
        )
        graph = eight_neighbor_graph(tiles)
        self.assertEqual(
            {edge.direction: edge.destination_index for edge in graph[0]},
            {"east": 1, "south": 2, "southeast": 3},
        )
        source = _mask((2, 5), (5, 3), (5, 5))

        east = rebase_tile_mask(source, tiles[0], tiles[1])
        south = rebase_tile_mask(source, tiles[0], tiles[2])
        southeast = rebase_tile_mask(source, tiles[0], tiles[3])

        np.testing.assert_array_equal(np.argwhere(east), [[2, 0], [5, 0]])
        np.testing.assert_array_equal(np.argwhere(south), [[0, 3], [0, 5]])
        np.testing.assert_array_equal(np.argwhere(southeast), [[0, 0]])
        np.testing.assert_array_equal(
            np.argwhere(rebase_tile_mask(_mask((2, 0)), tiles[1], tiles[0])),
            [[2, 5]],
        )
        lower = _mask((0, 3), (0, 5))
        np.testing.assert_array_equal(
            np.argwhere(rebase_tile_mask(lower, tiles[2], tiles[0])),
            [[5, 3], [5, 5]],
        )
        np.testing.assert_array_equal(
            np.argwhere(rebase_tile_mask(lower, tiles[2], tiles[1])),
            [[5, 0]],
        )

    def test_rebase_rejects_mismatched_mask_and_ignores_non_neighbors(self) -> None:
        tiles = plan_tile_grid(
            source_width=14,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        with self.assertRaisesRegex(ValueError, "does not match tile"):
            rebase_tile_mask(np.ones((5, 6), dtype=bool), tiles[0], tiles[1])
        self.assertFalse(
            rebase_tile_mask(np.ones((6, 6), dtype=bool), tiles[0], tiles[2]).any()
        )


if __name__ == "__main__":
    unittest.main()
