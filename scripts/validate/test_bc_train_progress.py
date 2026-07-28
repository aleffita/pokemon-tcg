"""Regression tests for MLX trainer progress and work accounting."""

import unittest

import numpy as np

from scripts.bc.bc_train_mlx import (
    _build_tbptt_groups,
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

    def test_invalid_work_divisor_is_rejected(self):
        with self.assertRaises(ValueError):
            _ceil_div(1, 0)


if __name__ == "__main__":
    unittest.main()
