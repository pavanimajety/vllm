# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection and rejection behavior for the split FlashInfer/TRTLLM backends.

FlashInfer and TRTLLM are separate backends serving different configuration
spaces. Which one auto-selection picks, and which configurations each refuses,
is the contract between them; these tests pin it.

Nothing here needs a GPU. Backend priorities are a pure function of the
capability tuple, and ``validate_configuration`` takes a synthetic
``DeviceCapability``, so every architecture is reachable from one host.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

SM90 = DeviceCapability(9, 0)
SM100 = DeviceCapability(10, 0)
SM103 = DeviceCapability(10, 3)
SM120 = DeviceCapability(12, 0)


@pytest.fixture(autouse=True)
def _clear_selection_caches():
    """Drop every cache that would leak one case's patches into the next."""
    cleared = []
    try:
        from vllm.platforms.cuda import _get_backend_priorities

        cleared.append(_get_backend_priorities)
    except ImportError:
        pass
    try:
        from vllm.utils.flashinfer import (
            has_nvidia_artifactory,
            supports_trtllm_attention,
        )

        cleared.extend([has_nvidia_artifactory, supports_trtllm_attention])
    except ImportError:
        pass

    for fn in cleared:
        fn.cache_clear()
    yield
    for fn in cleared:
        fn.cache_clear()


def _priorities(capability: DeviceCapability, *, use_mla=False, use_non_causal=False):
    _get_backend_priorities = pytest.importorskip(
        "vllm.platforms.cuda"
    )._get_backend_priorities
    return _get_backend_priorities(use_mla, capability, None, "auto", use_non_causal)


# ---------------------------------------------------------------------------
# Backend priority lists. Pure; no flashinfer import, no GPU.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability", [SM100, SM103], ids=["sm100", "sm103"])
def test_trtllm_outranks_flashinfer_on_blackwell_causal(capability):
    """TRTLLM leads the SM100-family causal list, with FlashInfer right behind.

    FlashInfer must stay directly behind it: TRTLLM rejects configurations it
    cannot serve, and selection falls through to the next entry.
    """
    priorities = _priorities(capability)
    assert priorities[0] is AttentionBackendEnum.TRTLLM
    assert priorities[1] is AttentionBackendEnum.FLASHINFER


def test_trtllm_absent_from_blackwell_non_causal():
    """Non-causal on SM100 prefers FlashAttention; TRTLLM cannot serve it."""
    priorities = _priorities(SM100, use_non_causal=True)
    assert AttentionBackendEnum.TRTLLM not in priorities
    assert priorities[0] is AttentionBackendEnum.FLASH_ATTN


@pytest.mark.parametrize("capability", [SM90, SM120], ids=["sm90", "sm120"])
def test_trtllm_absent_off_blackwell(capability):
    """TRTLLM serves only SM10x, so it must not be offered elsewhere."""
    priorities = _priorities(capability)
    assert AttentionBackendEnum.TRTLLM not in priorities
    assert AttentionBackendEnum.FLASHINFER in priorities


def test_trtllm_absent_from_mla_priorities():
    """The GQA split must not leak into the MLA priority lists."""
    assert AttentionBackendEnum.TRTLLM not in _priorities(SM100, use_mla=True)


# ---------------------------------------------------------------------------
# Configuration acceptance. Needs the backend classes, so needs flashinfer
# importable, but still no GPU.
# ---------------------------------------------------------------------------


def _fake_vllm_config(
    *,
    decode_context_parallel_size: int = 1,
    use_trtllm_attention: bool | None = None,
    disable_flashinfer_q_quantization: bool = False,
    num_attention_heads: int = 32,
    num_kv_heads: int = 8,
) -> SimpleNamespace:
    """Minimal stand-in exposing only what the backends' gates read."""
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=decode_context_parallel_size,
    )
    return SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _pc: num_attention_heads,
            get_num_kv_heads=lambda _pc: num_kv_heads,
        ),
        attention_config=SimpleNamespace(
            use_trtllm_attention=use_trtllm_attention,
            disable_flashinfer_q_quantization=disable_flashinfer_q_quantization,
        ),
    )


def _backends():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.attention.backends.trtllm import TRTLLMBackend

    return FlashInferBackend, TRTLLMBackend


def _reasons(
    backend_cls,
    *,
    capability: DeviceCapability,
    kv_cache_dtype="auto",
    block_size=16,
    has_sink=False,
    use_non_causal=False,
    head_size=128,
) -> list[str]:
    return backend_cls.validate_configuration(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        use_mla=False,
        has_sink=has_sink,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=capability,
        attn_type="decoder",
        use_non_causal=use_non_causal,
    )


@pytest.fixture
def trtllm_kernels_available(monkeypatch):
    """Pretend both TRTLLM kernels exist, independent of host and artifactory.

    Patched on the names ``trtllm.py`` imported, not at their definition site,
    and deliberately not on ``supports_trtllm_attention``'s callers elsewhere.
    """
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import trtllm as trtllm_backend

    monkeypatch.setattr(
        trtllm_backend, "supports_trtllm_attention", lambda is_prefill=False: True
    )
    monkeypatch.setattr(
        trtllm_backend, "can_use_trtllm_attention", lambda *a, **k: True
    )
    return trtllm_backend


@pytest.mark.parametrize(
    "capability,expected_ok",
    [(SM90, False), (SM100, True), (SM103, True), (SM120, False)],
    ids=["sm90", "sm100", "sm103", "sm120"],
)
def test_trtllm_compute_capability_gate(
    capability, expected_ok, trtllm_kernels_available
):
    """TRTLLM is SM10x-only: SM90 has XQA decode but no TRTLLM prefill kernel."""
    _, trtllm_cls = _backends()
    with set_current_vllm_config(_fake_vllm_config()):
        reasons = _reasons(trtllm_cls, capability=capability)
    if expected_ok:
        assert reasons == []
    else:
        assert "compute capability not supported" in reasons


@pytest.mark.parametrize("capability", [SM90, SM100, SM103, SM120])
def test_flashinfer_accepts_the_whole_supported_range(capability):
    """FlashInfer spans SM80..SM121, which is what makes it the fallback."""
    flashinfer_cls, _ = _backends()
    with set_current_vllm_config(_fake_vllm_config()):
        assert _reasons(flashinfer_cls, capability=capability) == []


@pytest.mark.parametrize(
    "capability",
    [DeviceCapability(7, 5), DeviceCapability(13, 0)],
    ids=["sm75", "sm130"],
)
def test_flashinfer_rejects_outside_its_range(capability):
    flashinfer_cls, _ = _backends()
    with set_current_vllm_config(_fake_vllm_config()):
        reasons = _reasons(flashinfer_cls, capability=capability)
    assert "compute capability not supported" in reasons


def test_dcp_rejects_trtllm_and_leaves_flashinfer(trtllm_kernels_available):
    """DCP is the fall-through case: TRTLLM refuses, FlashInfer takes it.

    TRTLLM prefill attends only the DCP-local KV shard and its decode cannot
    return LSE, so neither slice can be combined across ranks. Because TRTLLM
    leads the SM103 priority list, this rejection is what routes a DCP run to
    FlashInfer instead of failing.
    """
    flashinfer_cls, trtllm_cls = _backends()
    config = _fake_vllm_config(decode_context_parallel_size=2)
    with set_current_vllm_config(config):
        assert "decode context parallelism not supported" in _reasons(
            trtllm_cls, capability=SM103
        )
        assert _reasons(flashinfer_cls, capability=SM103) == []


def test_trtllm_rejects_unservable_head_ratio(monkeypatch):
    flashinfer_cls, trtllm_cls = _backends()
    from vllm.v1.attention.backends import trtllm as trtllm_backend

    monkeypatch.setattr(
        trtllm_backend, "supports_trtllm_attention", lambda is_prefill=False: True
    )
    monkeypatch.setattr(
        trtllm_backend, "can_use_trtllm_attention", lambda *a, **k: False
    )
    with set_current_vllm_config(_fake_vllm_config()):
        assert "TRTLLM decode does not support this query/KV head ratio" in _reasons(
            trtllm_cls, capability=SM103
        )


def test_trtllm_requires_both_prefill_and_decode_kernels(monkeypatch):
    """SM90's real shape: decode kernel present, prefill kernel absent."""
    _, trtllm_cls = _backends()
    from vllm.v1.attention.backends import trtllm as trtllm_backend

    monkeypatch.setattr(
        trtllm_backend,
        "supports_trtllm_attention",
        lambda is_prefill=False: not is_prefill,
    )
    with set_current_vllm_config(_fake_vllm_config()):
        assert "TRTLLM prefill and decode kernels are not both available" in _reasons(
            trtllm_cls, capability=SM103
        )


@pytest.mark.parametrize(
    "disable_q_quant,expect_rejected",
    [(True, True), (False, False)],
    ids=["q-quant-disabled", "q-quant-enabled"],
)
def test_trtllm_fp8_kv_depends_on_query_quantization(
    disable_q_quant, expect_rejected, trtllm_kernels_available
):
    """With a quantized KV cache, TRTLLM prefill is only auto-selected when the
    query is quantized to FP8 alongside it. ``build`` has no fallback, so the
    mismatch has to be caught at selection time."""
    _, trtllm_cls = _backends()
    config = _fake_vllm_config(disable_flashinfer_q_quantization=disable_q_quant)
    with set_current_vllm_config(config):
        reasons = _reasons(trtllm_cls, capability=SM103, kv_cache_dtype="fp8")
    rejected = any(r.startswith("TRTLLM prefill is not selected") for r in reasons)
    assert rejected is expect_rejected


def test_trtllm_large_block_size_forces_the_prefill_path(trtllm_kernels_available):
    """Pages >= 128 are served only by trtllm-gen, so they override the flag."""
    _, trtllm_cls = _backends()
    config = _fake_vllm_config(disable_flashinfer_q_quantization=True)
    with set_current_vllm_config(config):
        reasons = _reasons(
            trtllm_cls, capability=SM103, kv_cache_dtype="fp8", block_size=128
        )
    assert not any(r.startswith("TRTLLM prefill is not selected") for r in reasons)


def test_trtllm_rejects_non_causal_and_flashinfer_accepts_it(trtllm_kernels_available):
    flashinfer_cls, trtllm_cls = _backends()
    assert trtllm_cls.supports_non_causal() is False
    assert flashinfer_cls.supports_non_causal() is True
    with set_current_vllm_config(_fake_vllm_config()):
        assert "non-causal attention not supported" in _reasons(
            trtllm_cls, capability=SM103, use_non_causal=True
        )
        assert _reasons(flashinfer_cls, capability=SM103, use_non_causal=True) == []


def test_flashinfer_rejects_attention_sinks():
    """Sinks are why a gpt-oss-style model lands on TRTLLM on Blackwell."""
    flashinfer_cls, _ = _backends()
    assert flashinfer_cls.supports_sink() is False
    with set_current_vllm_config(_fake_vllm_config()):
        assert "attention sinks not supported" in _reasons(
            flashinfer_cls, capability=SM103, has_sink=True
        )


def test_flashinfer_does_not_advertise_nvfp4_kv_cache():
    """NVFP4 KV is TRTLLM-only; FlashInfer dropped the plumbing with the split."""
    flashinfer_cls, _ = _backends()
    assert "nvfp4" not in flashinfer_cls.supported_kv_cache_dtypes
    assert flashinfer_cls.supports_kv_cache_dtype("nvfp4") is False
