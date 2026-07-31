# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.eplb.metrics import EplbProm
from vllm.v1.metrics.stats import EplbMetricsStats


def test_physical_load_to_logical_sums_replicas():
    """Replica slots that share a logical expert are summed."""
    # 2 layers, 4 physical, 3 logical. Layer 0: logical 0 appears twice.
    expert_load = torch.tensor(
        [
            [10, 20, 30, 5],  # physical loads
            [1, 2, 3, 4],
        ],
        dtype=torch.int32,
    )
    physical_to_logical = torch.tensor(
        [
            [0, 1, 2, 0],  # physical 0 and 3 both map to logical 0
            [0, 1, 2, 1],
        ],
        dtype=torch.int64,
    )

    logical = EplbState.physical_load_to_logical(
        expert_load, physical_to_logical, num_logical_experts=3
    )

    assert logical.shape == (2, 3)
    assert logical.tolist() == [
        [15.0, 20.0, 30.0],  # 10+5, 20, 30
        [1.0, 6.0, 3.0],  # 1, 2+4, 3
    ]


def test_eplb_prom_observe_sets_expert_gauges():
    parallel_config = ParallelConfig(
        enable_eplb=True,
        eplb_config=EPLBConfig(prometheus_expert_load=True),
    )
    prom = EplbProm(
        parallel_config,
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["test-model", "0"]},
    )
    assert prom.enabled

    stats = EplbMetricsStats(
        tokens_per_logical_expert=[[10.0, 20.0], [3.0, 4.0]],
        rearrangements=1,
        last_rearrangement_seconds=0.5,
    )
    prom.observe(stats, engine_idx=0)

    assert set(prom._tokens_children) == {
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    }
    # Gauge values are readable via prometheus_client's public samples API.
    samples = {
        (s.labels["layer_idx"], s.labels["logical_expert_id"]): s.value
        for metric in prom._tokens_gauge.collect()
        for s in metric.samples
        if s.name == "vllm:eplb_tokens_routed_to_expert"
    }
    assert samples[("0", "0")] == 10.0
    assert samples[("0", "1")] == 20.0
    assert samples[("1", "0")] == 3.0
    assert samples[("1", "1")] == 4.0


def test_eplb_prom_appends_sparse_step_metrics_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "eplb_step_metrics.jsonl"
    monkeypatch.setenv("VLLM_EPLB_STEP_METRICS_PATH", str(path))
    parallel_config = ParallelConfig(
        enable_eplb=True,
        eplb_config=EPLBConfig(prometheus_expert_load=True),
    )
    prom = EplbProm(
        parallel_config,
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={3: ["test-model", "3"]},
    )

    stats = EplbMetricsStats(
        tokens_per_logical_expert=[[0.0, 2.0], [5.0, 0.0]],
    )
    prom.observe(stats, engine_idx=3, step_counter=17)

    layer0_records = [
        json.loads(line)
        for line in (tmp_path / "eplb_step_metrics" / "layer_000.jsonl")
        .read_text()
        .splitlines()
    ]
    layer1_records = [
        json.loads(line)
        for line in (tmp_path / "eplb_step_metrics" / "layer_001.jsonl")
        .read_text()
        .splitlines()
    ]
    assert layer0_records == [
        {
            "engine": 3,
            "step": 17,
            "layer": 0,
            "num_logical_experts": 2,
            "counts": [[1, 2]],
        }
    ]
    assert layer1_records == [
        {
            "engine": 3,
            "step": 17,
            "layer": 1,
            "num_logical_experts": 2,
            "counts": [[0, 5]],
        }
    ]


def test_eplb_prom_step_metrics_uses_monotonic_fallback(tmp_path, monkeypatch):
    path = tmp_path / "eplb_step_metrics.jsonl"
    monkeypatch.setenv("VLLM_EPLB_STEP_METRICS_PATH", str(path))
    parallel_config = ParallelConfig(
        enable_eplb=True,
        eplb_config=EPLBConfig(prometheus_expert_load=True),
    )
    prom = EplbProm(
        parallel_config,
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["test-model", "0"]},
    )

    stats = EplbMetricsStats(tokens_per_logical_expert=[[1.0]])
    prom.observe(stats, engine_idx=0, step_counter=0)
    prom.observe(stats, engine_idx=0, step_counter=0)
    prom.observe(stats, engine_idx=0, step_counter=0)

    records = [
        json.loads(line)
        for line in (tmp_path / "eplb_step_metrics" / "layer_000.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["step"] for record in records] == [0, 1, 2]


def test_eplb_prom_appends_token_expert_bitmap_sidecar(tmp_path, monkeypatch):
    path = tmp_path / "eplb_step_metrics.jsonl"
    monkeypatch.setenv("VLLM_EPLB_STEP_METRICS_PATH", str(path))
    parallel_config = ParallelConfig(
        enable_eplb=True,
        eplb_config=EPLBConfig(prometheus_expert_load=True),
    )
    prom = EplbProm(
        parallel_config,
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["test-model", "0"]},
    )

    # Four logical experts fit in one uint64 word. Token 0 hits experts 0/2;
    # token 1 hits experts 1/3.
    bitmap = (0b0101).to_bytes(8, "little") + (0b1010).to_bytes(8, "little")
    stats = EplbMetricsStats(
        tokens_per_logical_expert=[[1.0, 1.0, 1.0, 1.0]],
        token_expert_bitmaps=[bitmap],
    )
    prom.observe(stats, engine_idx=0, step_counter=4)

    layer_dir = tmp_path / "eplb_step_metrics"
    records = [
        json.loads(line)
        for line in (layer_dir / "layer_000.jsonl").read_text().splitlines()
    ]
    assert records == [
        {
            "engine": 0,
            "step": 4,
            "layer": 0,
            "num_logical_experts": 4,
            "counts": [[0, 1], [1, 1], [2, 1], [3, 1]],
            "token_expert_bitmap": {
                "path": "layer_000.token_expert_bitmap.u64",
                "offset_bytes": 0,
                "num_tokens": 2,
                "words_per_token": 1,
                "dtype": "uint64_le",
            },
        }
    ]
    assert (layer_dir / "layer_000.token_expert_bitmap.u64").read_bytes() == bitmap


def test_eplb_prom_step_metrics_reject_non_integral_counts(tmp_path, monkeypatch):
    path = tmp_path / "eplb_step_metrics.jsonl"
    monkeypatch.setenv("VLLM_EPLB_STEP_METRICS_PATH", str(path))
    parallel_config = ParallelConfig(
        enable_eplb=True,
        eplb_config=EPLBConfig(prometheus_expert_load=True),
    )
    prom = EplbProm(
        parallel_config,
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["test-model", "0"]},
    )

    stats = EplbMetricsStats(tokens_per_logical_expert=[[1.5]])
    with pytest.raises(ValueError, match="not integral"):
        prom.observe(stats, engine_idx=0, step_counter=1)


def test_eplb_prom_disabled_by_default():
    parallel_config = ParallelConfig(enable_eplb=True)
    prom = EplbProm(
        parallel_config,
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["test-model", "0"]},
    )
    assert not prom.enabled


def test_should_record_always_on_with_prometheus_expert_load():
    parallel_config = ParallelConfig(
        enable_eplb=True,
        eplb_config=EPLBConfig(
            prometheus_expert_load=True,
            window_size=10,
            step_interval=100,
        ),
    )
    state = EplbState(parallel_config, device=torch.device("cpu"))
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 100
    state.expert_load_window_size = 10

    assert state._should_record_current_step(log_stats=False) is True
