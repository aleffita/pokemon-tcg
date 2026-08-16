from __future__ import annotations

import torch

from rl.ropend import (
    apply_ropend,
    encoder_forward_ropend,
    temporal_coordinates,
    validate_ropend_config,
)


def test_apply_ropend_preserves_norm_and_identity_start() -> None:
    torch.manual_seed(7)
    tensor = torch.randn(2, 4, 9, 32)
    coordinates = torch.randn(2, 9, 3)
    scales = torch.zeros(3)
    identity = apply_ropend(tensor, coordinates, scales, (4, 4, 4))
    assert torch.equal(identity, tensor)

    rotated = apply_ropend(tensor, coordinates, torch.ones(3), (4, 4, 4))
    assert torch.isfinite(rotated).all()
    assert torch.allclose(
        rotated.square().sum(-1), tensor.square().sum(-1), atol=2e-5, rtol=2e-5
    )


def test_temporal_coordinates_anchor_recurrent_scratch() -> None:
    scalars = torch.zeros(2, 19)
    scalars[:, 0] = torch.tensor([0.2, 0.4])
    scalars[:, 1] = torch.tensor([0.25, 0.5])
    scalars[:, 17] = torch.tensor([0.2, 0.6])
    coordinates = temporal_coordinates(
        {"cls_scalars": scalars}, state_tokens=3, scratch_tokens=2, option_tokens=4
    )
    assert coordinates.shape == (2, 9, 3)
    assert coordinates[0, 0].tolist() == [10.0, 5.0, 1.0]
    assert torch.equal(coordinates[:, 3:5], torch.zeros(2, 2, 3))
    assert torch.equal(
        coordinates[:, -4:], coordinates[:, :1].expand(-1, 4, -1)
    )


def test_identity_start_matches_existing_encoder_and_axis_scales_train() -> None:
    torch.manual_seed(11)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=128,
        nhead=4,
        dim_feedforward=512,
        dropout=0.0,
        batch_first=True,
    )
    encoder = torch.nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
    encoder.eval()
    sequence = torch.randn(2, 13, 128)
    padding = torch.zeros(2, 13, dtype=torch.bool)
    padding[0, -3:] = True
    coordinates = torch.randn(2, 13, 3)
    scales = torch.nn.Parameter(torch.zeros(3))

    expected = encoder(sequence, src_key_padding_mask=padding)
    observed = encoder_forward_ropend(
        encoder, sequence, padding, coordinates, scales, (4, 4, 4)
    )
    assert torch.allclose(observed, expected, atol=2e-6, rtol=2e-6)

    target = torch.randn_like(observed)
    (observed * target).sum().backward()
    assert scales.grad is not None
    assert torch.isfinite(scales.grad).all()
    assert bool((scales.grad.abs() > 0).any())


def test_config_allocation_fits_stage4_head() -> None:
    config = validate_ropend_config(
        {
            "version": 1,
            "axes": ["turn", "logical_decision", "substep"],
            "pair_counts": [4, 4, 4],
            "base": 10_000.0,
            "init_scale": 0.0,
            "scratch_anchor": "zero",
        },
        head_dim=32,
    )
    assert config["rotary_dim"] == 24
