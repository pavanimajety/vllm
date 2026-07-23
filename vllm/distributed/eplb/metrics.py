# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

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

    def observe(self, eplb_stats: EplbMetricsStats, engine_idx: int = 0):
        if not self.enabled:
            return

        for layer_idx, expert_counts in enumerate(eplb_stats.tokens_per_logical_expert):
            for logical_expert_id, count in enumerate(expert_counts):
                self._get_tokens_child(engine_idx, layer_idx, logical_expert_id).set(
                    count
                )

        if eplb_stats.rearrangements > 0:
            self.counter_rearrangements[engine_idx].inc(eplb_stats.rearrangements)
            self.gauge_rearrangement_seconds[engine_idx].set(
                eplb_stats.last_rearrangement_seconds
            )
