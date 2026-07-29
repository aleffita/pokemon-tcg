"""Functional tests for FP16 training, optimizer routing, and sequential TBPTT."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.policy_mlx import build_token_net_mlx
from scripts.bc.bc_train_mlx import (
    _batched_sequential_tbptt_loss,
    _build_optimizer,
    _cross_entropy_sum,
    _sequential_tbptt_loss,
    _to_fp32_grads,
    _use_muon_parameter,
)
from scripts.validate.make_synthetic_data import make_dataset


def _optimizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="muon_adamw",
        lr=1e-3,
        muon_momentum=0.95,
        muon_weight_decay=0.01,
        adamw_betas=[0.9, 0.999],
        adamw_eps=1e-8,
        adamw_weight_decay=0.01,
    )


class _OptimizerFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(3, 3)
        self.card_emb = nn.Embedding(8, 3)
        self.opt_head = nn.Linear(3, 1)


class _RecurrentToy(nn.Module):
    """Minimal recurrent model whose memory is the cumulative input sum."""

    def __init__(self) -> None:
        super().__init__()
        self.scratch_tokens = 1
        self.d = 1
        self.learned_init = mx.zeros((1, 1), dtype=mx.float16)
        self.scale = mx.array([1.0], dtype=mx.float16)

    def logits_value(self, observation, memory_in=None):
        batch_size = observation["x"].shape[0]
        x = observation["x"].reshape(batch_size, 1, 1)
        if memory_in is None:
            memory_in = mx.zeros((batch_size, 1, 1), dtype=mx.float16)
        memory_out = memory_in + x * self.scale
        score = memory_out.reshape(batch_size)
        logits = mx.stack([score, -score], axis=-1)
        return logits, mx.zeros((batch_size,), dtype=mx.float16), memory_out


class BCTrainingSemanticTests(unittest.TestCase):
    def test_multi_optimizer_routes_only_hidden_matrices_to_muon(self):
        self.assertTrue(
            _use_muon_parameter(
                "encoder.layers.0.ff.layers.0.weight",
                mx.zeros((4, 4), dtype=mx.float16),
            )
        )
        self.assertFalse(
            _use_muon_parameter(
                "card_emb.weight", mx.zeros((8, 4), dtype=mx.float16)
            )
        )
        self.assertFalse(
            _use_muon_parameter(
                "value_head.weight", mx.zeros((1, 4), dtype=mx.float16)
            )
        )
        self.assertFalse(
            _use_muon_parameter("hidden.bias", mx.zeros((4,), dtype=mx.float16))
        )

    def test_optimizer_keeps_parameters_fp16_and_moments_fp32(self):
        model = _OptimizerFixture()
        model.set_dtype(mx.float16)
        optimizer = _build_optimizer(_optimizer_config())
        optimizer.init(model.trainable_parameters())
        grads = nn.utils.tree_map(
            lambda parameter: mx.ones(parameter.shape, dtype=mx.float32),
            model.trainable_parameters(),
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        self.assertTrue(
            all(
                parameter.dtype == mx.float16
                for _, parameter in nn.utils.tree_flatten(model.parameters())
            )
        )
        moment_leaves = [
            value
            for path, value in nn.utils.tree_flatten(optimizer.state)
            if path.endswith(".m") or path.endswith(".v")
        ]
        self.assertTrue(moment_leaves)
        self.assertTrue(all(value.dtype == mx.float32 for value in moment_leaves))

    def test_accumulated_loss_sums_normalize_exactly_once(self):
        model = nn.Linear(2, 2)
        model.set_dtype(mx.float16)
        features = mx.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=mx.float16
        )
        labels = mx.array([0, 1, 0], dtype=mx.int32)

        full_grad = mx.grad(
            lambda current: _cross_entropy_sum(current(features), labels)
        )(model)
        first_grad = mx.grad(
            lambda current: _cross_entropy_sum(
                current(features[:2]), labels[:2]
            )
        )(model)
        final_grad = mx.grad(
            lambda current: _cross_entropy_sum(
                current(features[2:]), labels[2:]
            )
        )(model)
        accumulated = nn.utils.tree_map(
            lambda first, final: (
                first.astype(mx.float32) + final.astype(mx.float32)
            )
            / 3,
            first_grad,
            final_grad,
        )
        expected = nn.utils.tree_map(
            lambda gradient: gradient.astype(mx.float32) / 3, full_grad
        )
        mx.eval(accumulated, expected)
        for (_, actual), (_, wanted) in zip(
            nn.utils.tree_flatten(accumulated),
            nn.utils.tree_flatten(expected),
        ):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(wanted), rtol=5e-4, atol=2e-4
            )

    def test_tbptt_unrolls_memory_sequentially_inside_chunk(self):
        model = _RecurrentToy()
        observations = {
            "x": mx.array([[1.0], [2.0], [3.0]], dtype=mx.float16)
        }
        labels = mx.array([0, 0, 0], dtype=mx.int32)

        (loss, memory_out), grads = mx.value_and_grad(
            _sequential_tbptt_loss, argnums=0
        )(model, observations, labels, None, [2, 1])
        grads = _to_fp32_grads(grads)
        mx.eval(loss, memory_out, grads)

        self.assertEqual(loss.dtype, mx.float32)
        self.assertEqual(memory_out.shape, (1, 1, 1))
        self.assertEqual(float(memory_out.item()), 5.0)
        self.assertTrue(
            all(
                gradient.dtype == mx.float32
                for _, gradient in nn.utils.tree_flatten(grads)
            )
        )

        def independent_loss(current):
            loss_sum = mx.array(0.0, dtype=mx.float32)
            for timestep in range(3):
                logits, _, _ = current.logits_value(
                    {"x": observations["x"][timestep : timestep + 1]},
                    memory_in=None,
                )
                loss_sum = loss_sum + _cross_entropy_sum(
                    logits, labels[timestep : timestep + 1]
                )
            return loss_sum

        independent_grads = _to_fp32_grads(mx.grad(independent_loss)(model))
        mx.eval(independent_grads)
        sequential_scale_grad = np.asarray(dict(nn.utils.tree_flatten(grads))["scale"])
        independent_scale_grad = np.asarray(
            dict(nn.utils.tree_flatten(independent_grads))["scale"]
        )
        self.assertFalse(
            np.allclose(sequential_scale_grad, independent_scale_grad)
        )
        _, _, lane_logits = _batched_sequential_tbptt_loss(
            model,
            [observations],
            [labels],
            [[2, 1]],
            [None],
            return_logits=True,
        )
        mx.eval(lane_logits)
        np.testing.assert_array_equal(
            np.asarray(lane_logits[0])[:, 0],
            np.asarray([1.0, 2.0, 5.0], dtype=np.float16),
        )

    def test_temporal_batch_keeps_episode_memories_isolated(self):
        model = _RecurrentToy()
        lane_observations = [
            {"x": mx.array([[1.0], [2.0]], dtype=mx.float16)},
            {"x": mx.array([[10.0], [20.0]], dtype=mx.float16)},
        ]
        lane_labels = [
            mx.array([0, 0], dtype=mx.int32),
            mx.array([0, 0], dtype=mx.int32),
        ]
        loss, memory, lane_logits = _batched_sequential_tbptt_loss(
            model,
            lane_observations,
            lane_labels,
            [[1, 1], [1, 1]],
            [None, None],
            return_logits=True,
        )
        mx.eval(loss, memory, lane_logits)

        np.testing.assert_array_equal(
            np.asarray(memory).reshape(-1),
            np.asarray([3.0, 30.0], dtype=np.float16),
        )
        np.testing.assert_array_equal(
            np.asarray(lane_logits[0])[:, 0],
            np.asarray([1.0, 3.0], dtype=np.float16),
        )
        np.testing.assert_array_equal(
            np.asarray(lane_logits[1])[:, 0],
            np.asarray([10.0, 30.0], dtype=np.float16),
        )

    def test_real_policy_forward_is_fp16_with_sixteen_registers(self):
        with tempfile.TemporaryDirectory(prefix="ptcg_mlx_fp16_") as out:
            make_dataset(1, out, seed=11)
            arrays = {
                path.stem: np.load(path)
                for path in Path(out).glob("*.npy")
                if not path.name.startswith("__")
            }
        arrays["effect_mask"] = np.zeros((1, 2), dtype=np.float16)
        arrays["action_mask"][:] = 0
        arrays["action_mask"][0, [0, 150, 192]] = 1

        card_table = get_card_table()
        int_keys = set(TokenEncoder(card_table).int_keys)
        observation = {
            key: mx.array(
                value.astype(np.int32 if key in int_keys else np.float16)
            )
            for key, value in arrays.items()
        }
        model = build_token_net_mlx(
            card_table,
            {
                "d_model": 128,
                "nhead": 4,
                "nlayers": 1,
                "static": True,
                "split_heads": True,
                "structured": True,
                "scratch_registers": 16,
            },
        )
        model.set_dtype(mx.float16)
        logits, value, memory = model.logits_value(observation)
        mx.eval(logits, value, memory)

        self.assertEqual(logits.dtype, mx.float16)
        self.assertEqual(value.dtype, mx.float16)
        self.assertEqual(memory.dtype, mx.float16)
        self.assertEqual(memory.shape, (1, 16, 128))
        self.assertTrue(np.isfinite(np.asarray(logits)).all())
        self.assertTrue(np.isfinite(np.asarray(logits)[0, 150]))


if __name__ == "__main__":
    unittest.main()
