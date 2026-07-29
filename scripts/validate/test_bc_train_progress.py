"""Regression tests for MLX trainer progress and work accounting."""

import unittest

import numpy as np

from scripts.bc.bc_train_mlx import (
    _build_tbptt_decision_groups,
    _build_tbptt_groups,
    _build_tbptt_plan,
    _ceil_div,
    _standard_microbatch_count,
    _tbptt_microbatch_count,
)


class TrainingProgressAccountingTests(unittest.TestCase):
    def test_standard_batches_include_partial_batches_at_each_slab_boundary(self):
        slab_bounds = [(0, 5), (5, 8)]

        self.assertEqual(_standard_microbatch_count(slab_bounds, batch_size=4), 3)

    def test_tbptt_count_matches_interleaved_episode_side_groups(self):
        episode_meta = np.array(
            [
                ("20", 0),
                ("10", 1),
                ("20", 0),
                ("10", 1),
                ("20", 0),
            ],
            dtype=[("episode_id", "U64"), ("side", "i4")],
        )

        groups = _build_tbptt_groups(episode_meta, train_rows=len(episode_meta))

        self.assertEqual([rows.tolist() for rows in groups], [[1, 3], [0, 2, 4]])
        self.assertEqual(_tbptt_microbatch_count(groups, chunk_size=2), 3)

    def test_optimizer_steps_include_partial_accumulation_window(self):
        self.assertEqual(_ceil_div(5, 2), 3)
        self.assertEqual(_ceil_div(6, 2), 3)

    def test_tbptt_never_splits_decisions_and_batch_budget_changes_work(self):
        episode_meta = np.array(
            [
                ("episode-a", 0, 0),
                ("episode-a", 0, 0),
                ("episode-a", 0, 1),
                ("episode-b", 0, 0),
                ("episode-b", 0, 1),
                ("episode-b", 0, 1),
            ],
            dtype=[("episode_id", "U64"), ("side", "i4"), ("step_id", "i4")],
        )
        groups = _build_tbptt_decision_groups(
            episode_meta, train_rows=len(episode_meta)
        )

        self.assertEqual(
            [[rows.tolist() for rows in group] for group in groups],
            [[[0, 1], [2]], [[3], [4, 5]]],
        )
        small_plan = _build_tbptt_plan(groups, chunk_size=1, row_budget=2)
        large_plan = _build_tbptt_plan(groups, chunk_size=1, row_budget=4)

        self.assertEqual(len(small_plan), 4)
        self.assertEqual(len(large_plan), 2)
        self.assertEqual(_ceil_div(len(small_plan), 2), 2)
        self.assertEqual(_ceil_div(len(large_plan), 2), 1)
        self.assertTrue(
            all(
                len({chunk.group_index for chunk in temporal_batch})
                == len(temporal_batch)
                for temporal_batch in large_plan
            )
        )

    def test_invalid_work_divisor_is_rejected(self):
        with self.assertRaises(ValueError):
            _ceil_div(1, 0)

    def test_large_temporal_chunk_is_split_at_decision_boundaries(self):
        plan = _build_tbptt_plan(
            [[np.arange(0, 70), np.arange(70, 140)]],
            chunk_size=32,
            row_budget=128,
        )

        self.assertEqual(
            [sum(chunk.row_count for chunk in batch) for batch in plan],
            [70, 70],
        )
        self.assertTrue(plan[0][0].is_new_group)
        self.assertFalse(plan[1][0].is_new_group)

    def test_indivisible_decision_over_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one engine decision exceeds"):
            _build_tbptt_plan(
                [[np.arange(129)]],
                chunk_size=32,
                row_budget=128,
            )


if __name__ == "__main__":
    unittest.main()
