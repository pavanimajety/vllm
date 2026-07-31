# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
import os
import threading

import prometheus_client

from vllm.config import ParallelConfig
from vllm.v1.metrics.stats import EplbMetricsStats


def _make_per_engine(
    metric: prometheus_client.Gauge | prometheus_client.Counter,
    per_engine_labelvalues: dict[int, list[object]],
) -> dict[int, prometheus_client.Gauge | prometheus_client.Counter]:
    return {
        idx: metric.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }


class EplbProm:
    """Record opt-in per-logical-expert EPLB load metrics in Prometheus.

    Each engine independently reports how many tokens it routed to each
    logical expert, per MoE layer, for the current engine step. Replica
    slots that share a logical expert are summed before export. Gauges
    are summable across DP ranks:

        sum by (layer_idx, logical_expert_id) (
            vllm:eplb_tokens_routed_to_expert
        )

    Enabled only when ``EPLBConfig.prometheus_expert_load`` is true.
    """

    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        parallel_config: ParallelConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.enabled = (
            parallel_config.enable_eplb
            and parallel_config.eplb_config.prometheus_expert_load
        )
        if not self.enabled:
            return

        extended_labels = [*labelnames, "layer_idx", "logical_expert_id"]
        self._tokens_gauge = self._gauge_cls(
            name="vllm:eplb_tokens_routed_to_expert",
            documentation=(
                "Tokens routed to each logical expert per MoE layer for the "
                "current engine step (from this DP rank's perspective). "
                "Replica counts for the same logical expert are summed. "
                "Summable across DP ranks."
            ),
            multiprocess_mode="mostrecent",
            labelnames=extended_labels,
        )
        self._per_engine_labelvalues = per_engine_labelvalues

        counter_rearrangements = self._counter_cls(
            name="vllm:eplb_rearrangements_total",
            documentation="Total number of EPLB expert rearrangements.",
            labelnames=labelnames,
        )
        self.counter_rearrangements = _make_per_engine(
            counter_rearrangements, per_engine_labelvalues
        )

        gauge_rearrangement_seconds = self._gauge_cls(
            name="vllm:eplb_rearrangement_seconds",
            documentation=(
                "Duration of the most recent EPLB expert rearrangement in seconds."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_rearrangement_seconds = _make_per_engine(
            gauge_rearrangement_seconds, per_engine_labelvalues
        )

        self._tokens_children: dict[
            tuple[int, int, int],
            prometheus_client.Gauge,
        ] = {}
        self._step_metrics_dir = os.environ.get("VLLM_EPLB_STEP_METRICS_DIR")
        step_metrics_path = os.environ.get("VLLM_EPLB_STEP_METRICS_PATH")
        if self._step_metrics_dir is None and step_metrics_path is not None:
            if step_metrics_path.endswith(".jsonl"):
                self._step_metrics_dir = os.path.splitext(step_metrics_path)[0]
            else:
                self._step_metrics_dir = step_metrics_path
        self._step_metrics_lock = threading.Lock()
        self._step_metrics_counters: dict[int, int] = {}
        self._step_metrics_fds: dict[int, int] = {}
        self._token_bitmap_fds: dict[int, int] = {}
        self._token_bitmap_offsets: dict[int, int] = {}

    def _get_tokens_child(
        self, engine_idx: int, layer_idx: int, logical_expert_id: int
    ) -> prometheus_client.Gauge:
        key = (engine_idx, layer_idx, logical_expert_id)
        child = self._tokens_children.get(key)
        if child is None:
            base_labels = self._per_engine_labelvalues[engine_idx]
            child = self._tokens_gauge.labels(
                *base_labels, str(layer_idx), str(logical_expert_id)
            )
            self._tokens_children[key] = child
        return child

    @staticmethod
    def _integer_count(count: float) -> int:
        count_float = float(count)
        if not count_float.is_integer():
            raise ValueError(f"EPLB routed-token count is not integral: {count}")
        return int(count_float)

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]

    def _append_step_metrics(
        self,
        eplb_stats: EplbMetricsStats,
        engine_idx: int,
        step_counter: int | None,
    ) -> None:
        if self._step_metrics_dir is None:
            return

        with self._step_metrics_lock:
            previous_step = self._step_metrics_counters.get(engine_idx)
            if previous_step is None:
                output_step = 0 if step_counter is None else step_counter
            elif step_counter is None or step_counter <= previous_step:
                output_step = previous_step + 1
            else:
                output_step = step_counter
            self._step_metrics_counters[engine_idx] = output_step

        os.makedirs(self._step_metrics_dir, exist_ok=True)
        num_logical_experts = (
            len(eplb_stats.tokens_per_logical_expert[0])
            if eplb_stats.tokens_per_logical_expert
            else 0
        )
        words_per_token = math.ceil(num_logical_experts / 64)
        with self._step_metrics_lock:
            for layer_idx, expert_counts in enumerate(
                eplb_stats.tokens_per_logical_expert
            ):
                nonzero_counts: list[list[int]] = []
                for logical_expert_id, count in enumerate(expert_counts):
                    int_count = self._integer_count(count)
                    if int_count:
                        nonzero_counts.append([logical_expert_id, int_count])
                record = {
                    "engine": engine_idx,
                    "step": output_step,
                    "layer": layer_idx,
                    "num_logical_experts": num_logical_experts,
                    "counts": nonzero_counts,
                }
                if eplb_stats.token_expert_bitmaps is not None:
                    bitmap = eplb_stats.token_expert_bitmaps[layer_idx]
                    if bitmap is not None:
                        bytes_per_token = words_per_token * 8
                        if bytes_per_token == 0 or len(bitmap) % bytes_per_token:
                            raise ValueError(
                                "EPLB token-expert bitmap bytes are not aligned: "
                                f"layer={layer_idx}, bytes={len(bitmap)}, "
                                f"bytes_per_token={bytes_per_token}"
                            )
                        bitmap_fd = self._token_bitmap_fds.get(layer_idx)
                        if bitmap_fd is None:
                            bitmap_fd = os.open(
                                os.path.join(
                                    self._step_metrics_dir,
                                    (f"layer_{layer_idx:03d}.token_expert_bitmap.u64"),
                                ),
                                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                                0o644,
                            )
                            self._token_bitmap_fds[layer_idx] = bitmap_fd
                            self._token_bitmap_offsets[layer_idx] = 0
                        offset = self._token_bitmap_offsets[layer_idx]
                        self._write_all(bitmap_fd, bitmap)
                        self._token_bitmap_offsets[layer_idx] = offset + len(bitmap)
                        record["token_expert_bitmap"] = {
                            "path": f"layer_{layer_idx:03d}.token_expert_bitmap.u64",
                            "offset_bytes": offset,
                            "num_tokens": len(bitmap) // bytes_per_token,
                            "words_per_token": words_per_token,
                            "dtype": "uint64_le",
                        }
                line = json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
                fd = self._step_metrics_fds.get(layer_idx)
                if fd is None:
                    fd = os.open(
                        os.path.join(
                            self._step_metrics_dir, f"layer_{layer_idx:03d}.jsonl"
                        ),
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                        0o644,
                    )
                    self._step_metrics_fds[layer_idx] = fd
                self._write_all(fd, line)

    def observe(
        self,
        eplb_stats: EplbMetricsStats,
        engine_idx: int = 0,
        step_counter: int | None = None,
    ):
        if not self.enabled:
            return

        for layer_idx, expert_counts in enumerate(eplb_stats.tokens_per_logical_expert):
            for logical_expert_id, count in enumerate(expert_counts):
                self._get_tokens_child(engine_idx, layer_idx, logical_expert_id).set(
                    count
                )

        self._append_step_metrics(eplb_stats, engine_idx, step_counter)

        if eplb_stats.rearrangements > 0:
            self.counter_rearrangements[engine_idx].inc(eplb_stats.rearrangements)
            self.gauge_rearrangement_seconds[engine_idx].set(
                eplb_stats.last_rearrangement_seconds
            )
