from __future__ import annotations

import pytest
import torch

from statarb.neural.networks import ARCHITECTURES, make_policy


def features(batch: int = 5, lookback: int = 30, channels: int = 2) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(7)
    return torch.randn((batch, lookback, channels), generator=generator)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_shape_bounds_and_finite_gradients(architecture: str) -> None:
    policy = make_policy(architecture)
    x = features().requires_grad_(True)
    preference = policy(x)
    assert preference.shape == (len(x),)
    assert torch.isfinite(preference).all()
    assert (preference.abs() <= 1).all()
    preference.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    gradients = [parameter.grad for parameter in policy.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients if gradient is not None)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_initialization_is_deterministic_and_restores_cpu_rng(architecture: str) -> None:
    before = torch.get_rng_state().clone()
    first = make_policy(architecture, seed=121)
    assert torch.equal(before, torch.get_rng_state())
    second = make_policy(architecture, seed=121)

    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    different = make_policy(architecture, seed=122)
    assert any(
        not torch.equal(value, different.state_dict()[name])
        for name, value in first.state_dict().items()
    )


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("training", [False, True])
def test_prediction_does_not_depend_on_other_batch_members_or_prior_calls(
    architecture: str, training: bool
) -> None:
    policy = make_policy(architecture)
    policy.train(training)
    x = features()
    before = torch.get_rng_state().clone()
    together = policy(x)
    alone = policy(x[:1])
    policy(features(batch=3) * 20)
    repeated = policy(x[:1])
    torch.testing.assert_close(together[:1], alone, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(alone, repeated, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("hidden_size", [1, 7, 12])
def test_configured_dimensions_and_empty_batch(architecture: str, hidden_size: int) -> None:
    policy = make_policy(architecture, lookback=9, channels=3, hidden_size=hidden_size)
    assert policy(features(batch=2, lookback=9, channels=3)).shape == (2,)
    assert policy(features(batch=0, lookback=9, channels=3)).shape == (0,)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_rejects_wrong_shape_and_integer_inputs(architecture: str) -> None:
    policy = make_policy(architecture)

    with pytest.raises(ValueError, match="shape"):
        policy(torch.zeros((2, 29, 2)))

    with pytest.raises(ValueError, match="shape"):
        policy(torch.zeros((30, 2)))

    with pytest.raises(ValueError, match="floating"):
        policy(torch.zeros((2, 30, 2), dtype=torch.int64))


@pytest.mark.parametrize("kwargs", [{"lookback": 0}, {"channels": -1}, {"hidden_size": True}])
def test_rejects_invalid_dimensions(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        make_policy("lstm", **kwargs)


def test_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        make_policy("autoencoder")
