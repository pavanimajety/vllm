# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import json
import threading
import types
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RouterTopKBitmapDumper,
    _resolve_writer_ranks,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class _DummyRouter:
        _routing_replay_out: torch.Tensor | None = None

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self._quant_method = self.routed_experts.quant_method

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module])
    )

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self._quant_method = self.routed_experts.quant_method

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module])
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


def test_gpu_model_runner_does_not_bind_draft_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self._quant_method = self.routed_experts.quant_method

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [target_module]),
        compilation_config=types.SimpleNamespace(
            static_forward_context={
                "model.layers.7.mlp.experts": target_module,
                "mtp.layers.0.mlp.experts": draft_module,
            }
        ),
    )

    capturer = types.SimpleNamespace(capture=lambda *_: None)
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None


def test_gpu_model_runner_rejects_monolithic_without_replay_support(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            # Use a concrete monolithic expert and override its capability
            # instead of instantiating the abstract base class directly.
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsFp8,
            )

            fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
            self.routed_experts = types.SimpleNamespace(
                quant_method=types.SimpleNamespace(
                    is_monolithic=True,
                    moe_kernel=types.SimpleNamespace(
                        impl=types.SimpleNamespace(fused_experts=fused_experts)
                    ),
                )
            )
            self._quant_method = self.routed_experts.quant_method
            self._quant_method.moe_kernel.impl.fused_experts = fused_experts
            fused_experts.supports_routing_replay_capture = lambda: False

    class DummyCapturer:
        def capture(self, layer_id, topk_ids):
            pass

    dummy_module = DummyFusedMoE()
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module])
    )

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, DummyCapturer())


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def _dumper_for_cpu_test(tmp_path, rank=0, writer_ranks=None):
    dumper = RouterTopKBitmapDumper.__new__(RouterTopKBitmapDumper)
    dumper.output_dir = tmp_path
    dumper.dp_rank = 0
    dumper.tp_size = 1
    dumper.rank = rank
    dumper.writer_ranks = writer_ranks
    dumper.is_writer = writer_ranks is None or rank in writer_ranks
    dumper.local_rank = 0
    dumper.physical_gpu_id = 3
    dumper.hostname = "host/a"
    dumper.role = "decode"
    dumper._lock = threading.Lock()
    dumper._step_by_layer = defaultdict(int)
    dumper._engine_iteration = None
    dumper._active_engine_iteration = None
    dumper._active_request_token_count = None
    dumper._captured_layers = set()
    dumper._request_spans = []
    dumper._session_request_ordinals = defaultdict(dict)
    dumper._next_session_request_ordinal = defaultdict(int)
    return dumper


def test_router_topk_bitmap_dumper_serializes_request_spans(tmp_path):
    dumper = _dumper_for_cpu_test(tmp_path)
    dumper.set_engine_iteration(7)
    dumper.set_request_spans(
        ["internal-a", "internal-b"],
        [2, 1],
        {
            "internal-a": {
                "x-request-id": "request-a",
                "x-correlation-id": "correlation-a",
            },
            "internal-b": {
                "x-request-id": "request-b",
            },
        },
    )

    with patch(
        f"{_REC_MODULE}.get_forward_context",
        return_value=SimpleNamespace(dp_metadata=None),
    ):
        dumper.capture(
            layer_id=4,
            topk_ids=torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64),
            num_logical_experts=8,
        )

    jsonl_path = (
        tmp_path
        / "role_decode.host_host_a.gpu_003.rank_000.local_000.dp_000.layer_004.jsonl"
    )
    stem = dumper._file_stem(4)
    assert stem.count(".gpu_") == 1
    assert stem == (
        "role_decode.host_host_a.gpu_003.rank_000.local_000.dp_000.layer_004"
    )
    record = json.loads(jsonl_path.read_text().strip())
    assert record["rank"] == 0
    assert record["local_rank"] == 0
    assert record["gpu_id"] == 3
    assert record["hostname"] == "host/a"
    assert record["dp_rank"] == 0
    assert record["role"] == "decode"
    assert record["step"] == 7
    assert record["num_tokens"] == 3
    assert sum(count for _, count in record["counts"]) == 6
    bitmap_path = tmp_path / record["token_expert_bitmap"]["path"]
    assert bitmap_path.stat().st_size == 3 * 8
    assert record["token_expert_bitmap"]["offset_bytes"] == 0
    assert record["token_expert_bitmap"]["num_tokens"] == 3
    assert record["token_expert_bitmap"]["words_per_token"] == 1
    assert record["request_spans"] == [
        {
            "request_id": "request-a",
            "internal_request_id": "internal-a",
            "correlation_id": "correlation-a",
            "session_id": "correlation-a",
            "turn_id": "request-a",
            "turn_id_source": "x-request-id",
            "session_request_ordinal": 0,
            "token_start": 0,
            "token_count": 2,
        },
        {
            "request_id": "request-b",
            "internal_request_id": "internal-b",
            "correlation_id": None,
            "session_id": None,
            "turn_id": "internal-b",
            "turn_id_source": "internal_request_id_fallback",
            "session_request_ordinal": None,
            "token_start": 2,
            "token_count": 1,
        },
    ]


def test_router_topk_bitmap_dumper_writer_ranks_env_parsing():
    assert _resolve_writer_ranks(None) is None
    assert _resolve_writer_ranks("") is None
    assert _resolve_writer_ranks("   ") is None
    assert _resolve_writer_ranks("0") == frozenset({0})
    assert _resolve_writer_ranks(" 0, 4 ,8 ") == frozenset({0, 4, 8})
    for invalid in ("abc", "-1", ","):
        with pytest.raises(RuntimeError):
            _resolve_writer_ranks(invalid)


def test_router_topk_bitmap_dumper_non_writer_rank_writes_nothing(tmp_path):
    """A gated-out rank must stay silent on disk but still look "captured".

    Under TP every rank sees the same top-k IDs, so gating writers removes
    duplicate payloads. The rank must still mark the layer, otherwise
    ``set_engine_iteration``'s zero-capture guard would fire on the next step.
    """
    dumper = _dumper_for_cpu_test(tmp_path, rank=3, writer_ranks=frozenset({0}))
    dumper.set_engine_iteration(1)
    dumper.set_request_spans(["internal-a"], [3], {})

    with patch(
        f"{_REC_MODULE}.get_forward_context",
        return_value=SimpleNamespace(dp_metadata=None),
    ):
        dumper.capture(
            layer_id=4,
            topk_ids=torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64),
            num_logical_experts=8,
        )

    assert list(tmp_path.iterdir()) == []
    assert dumper._captured_layers == {4}
    # The guard must not trip now that the layer is marked.
    dumper.set_engine_iteration(2)


def test_router_topk_bitmap_dumper_selected_writer_rank_still_writes(tmp_path):
    dumper = _dumper_for_cpu_test(tmp_path, rank=4, writer_ranks=frozenset({0, 4}))
    dumper.set_engine_iteration(1)
    dumper.set_request_spans(["internal-a"], [3], {})

    with patch(
        f"{_REC_MODULE}.get_forward_context",
        return_value=SimpleNamespace(dp_metadata=None),
    ):
        dumper.capture(
            layer_id=4,
            topk_ids=torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64),
            num_logical_experts=8,
        )

    jsonl_path = tmp_path / f"{dumper._file_stem(4)}.jsonl"
    record = json.loads(jsonl_path.read_text().strip())
    assert record["rank"] == 4
    assert record["num_tokens"] == 3
    assert dumper._captured_layers == {4}


def test_router_topk_bitmap_dumper_clips_local_dp_spans():
    dumper = _dumper_for_cpu_test(None)
    dumper.set_request_spans(
        ["request-a", "request-b"],
        [2, 1],
        {},
    )

    # start/end are global offsets, but the request list is local to the
    # captured DP rank. The dumper should keep local token offsets.
    assert dumper._slice_request_spans(2, 5) == [
        {
            "request_id": "request-a",
            "internal_request_id": "request-a",
            "correlation_id": None,
            "session_id": None,
            "turn_id": "request-a",
            "turn_id_source": "internal_request_id_fallback",
            "session_request_ordinal": None,
            "token_start": 0,
            "token_count": 2,
        },
        {
            "request_id": "request-b",
            "internal_request_id": "request-b",
            "correlation_id": None,
            "session_id": None,
            "turn_id": "request-b",
            "turn_id_source": "internal_request_id_fallback",
            "session_request_ordinal": None,
            "token_start": 2,
            "token_count": 1,
        },
    ]


def test_router_topk_bitmap_dumper_assigns_worker_local_session_ordinals():
    dumper = _dumper_for_cpu_test(None)
    dumper.set_request_spans(
        ["internal-a", "internal-b"],
        [1, 1],
        {
            "internal-a": {
                "x-request-id": "request-a",
                "x-correlation-id": "session-a",
            },
            "internal-b": {
                "x-request-id": "request-b",
                "x-correlation-id": "session-a",
            },
        },
    )
    assert [span["session_request_ordinal"] for span in dumper._request_spans] == [
        0,
        1,
    ]

    # The same request can be scheduled for multiple decode executions and
    # must retain its ordinal.
    dumper.set_request_spans(
        ["internal-a"],
        [1],
        {
            "internal-a": {
                "x-request-id": "request-a",
                "x-correlation-id": "session-a",
            }
        },
    )
    assert dumper._request_spans[0]["session_request_ordinal"] == 0

    # A new session starts its own ordinal sequence.
    dumper.set_request_spans(
        ["internal-c", "internal-d"],
        [1, 1],
        {
            "internal-c": {
                "x-request-id": "request-c",
                "x-correlation-id": "session-b",
            },
            "internal-d": {
                "x-request-id": "request-d",
                "x-correlation-id": "session-b",
            },
        },
    )
    assert [span["session_request_ordinal"] for span in dumper._request_spans] == [
        0,
        1,
    ]


def test_router_topk_bitmap_dumper_turn_id_fallback_and_case_insensitive_headers():
    dumper = _dumper_for_cpu_test(None)
    dumper.set_request_spans(
        ["internal-a", "internal-b"],
        [1, 1],
        {
            "internal-a": {
                "X-Request-ID": "request-a",
                "X-Correlation-ID": "session-a",
            },
            "internal-b": {"x-correlation-id": "session-a"},
        },
    )
    first, second = dumper._request_spans
    assert first["session_id"] == "session-a"
    assert first["turn_id"] == "request-a"
    assert first["turn_id_source"] == "x-request-id"
    assert first["session_request_ordinal"] == 0
    assert second["session_id"] == "session-a"
    assert second["turn_id"] == "internal-b"
    assert second["turn_id_source"] == "internal_request_id_fallback"
    assert second["session_request_ordinal"] == 1


def test_router_topk_bitmap_dumper_rejects_unresolved_writer_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("SLURM_LOCALID", raising=False)
    monkeypatch.setenv("VLLM_ROUTER_TOPK_BITMAP_ROLE", "decode")
    monkeypatch.setattr(f"{_REC_MODULE}._resolve_physical_gpu_id", lambda: 3)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            tensor_parallel_size=1,
        )
    )
    with pytest.raises(RuntimeError, match="LOCAL_RANK or SLURM_LOCALID"):
        RouterTopKBitmapDumper(str(tmp_path), config)


def test_router_topk_bitmap_dumper_zero_capture_guard():
    dumper = _dumper_for_cpu_test(None)
    dumper.set_engine_iteration(0)
    dumper.set_request_spans(["internal-a"], [1], {})
    with pytest.raises(RuntimeError, match="no router capture occurred"):
        dumper.set_engine_iteration(1)


class _BitmapDumperForBindingTest:
    def __init__(self):
        self.calls = []

    def capture(self, layer_id, topk_ids, num_logical_experts):
        self.calls.append((layer_id, topk_ids, num_logical_experts))


def _bitmap_binding_module(*, quant_method, router=None):
    return types.SimpleNamespace(
        layer_id=7,
        router=router or _make_router(),
        moe_config=SimpleNamespace(num_logical_experts=16),
        _quant_method=quant_method,
    )


@pytest.mark.parametrize(
    "worker_module_name",
    ["vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"],
)
def test_bitmap_dumper_binds_modular_router_in_both_workers(
    monkeypatch, worker_module_name
):
    worker_module = importlib.import_module(worker_module_name)
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    router = _make_router()
    previous_calls = []
    router.set_capture_fn(lambda ids: previous_calls.append(ids.clone()))
    module = _bitmap_binding_module(
        quant_method=types.SimpleNamespace(is_monolithic=False),
        router=router,
    )
    monkeypatch.setattr(fused_moe_layer, "MoERunner", type(module))
    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [module])
    )
    dumper = _BitmapDumperForBindingTest()

    worker_module.GPUModelRunner._bind_router_topk_bitmap_dumper(dummy_self, dumper)
    router.capture_fn(torch.tensor([[5, 6]]))

    assert len(previous_calls) == 1
    assert len(dumper.calls) == 1
    assert dumper.calls[0][0] == 7
    assert dumper.calls[0][2] == 16


def _monolithic_bitmap_module(*, supports_replay):
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import CPUExpertsFp8

    fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
    previous_calls = []
    fused_experts.routing_replay_capture_fn = lambda ids: previous_calls.append(
        ids.clone()
    )
    fused_experts.supports_routing_replay_capture = lambda: supports_replay
    fused_experts.set_capture_fn = lambda fn: setattr(
        fused_experts, "routing_replay_capture_fn", fn
    )
    module = _bitmap_binding_module(
        quant_method=types.SimpleNamespace(
            is_monolithic=True,
            moe_kernel=types.SimpleNamespace(
                impl=types.SimpleNamespace(fused_experts=fused_experts)
            ),
        )
    )
    module._test_previous_calls = previous_calls
    module._test_fused_experts = fused_experts
    return module


def test_bitmap_dumper_binds_supported_monolithic_router(monkeypatch):
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
    from vllm.v1.worker import gpu_model_runner as gmr

    module = _monolithic_bitmap_module(supports_replay=True)
    monkeypatch.setattr(fused_moe_layer, "MoERunner", type(module))
    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [module])
    )
    dumper = _BitmapDumperForBindingTest()

    gmr.GPUModelRunner._bind_router_topk_bitmap_dumper(dummy_self, dumper)
    module._test_fused_experts.routing_replay_capture_fn(torch.tensor([[5, 6]]))

    assert len(module._test_previous_calls) == 1
    assert len(dumper.calls) == 1
    assert dumper.calls[0][0] == 7


def test_bitmap_dumper_rejects_unsupported_monolithic_router(monkeypatch):
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
    from vllm.v1.worker import gpu_model_runner as gmr

    module = _monolithic_bitmap_module(supports_replay=False)
    monkeypatch.setattr(fused_moe_layer, "MoERunner", type(module))
    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [module])
    )

    with pytest.raises(ValueError, match="unsupported monolithic MoE kernel"):
        gmr.GPUModelRunner._bind_router_topk_bitmap_dumper(
            dummy_self, _BitmapDumperForBindingTest()
        )


def test_bitmap_dumper_rejects_zero_binding():
    from vllm.v1.worker import gpu_model_runner as gmr

    dummy_self = types.SimpleNamespace(model=types.SimpleNamespace(modules=lambda: []))
    with pytest.raises(RuntimeError, match="zero MoE capture targets"):
        gmr.GPUModelRunner._bind_router_topk_bitmap_dumper(
            dummy_self, _BitmapDumperForBindingTest()
        )
