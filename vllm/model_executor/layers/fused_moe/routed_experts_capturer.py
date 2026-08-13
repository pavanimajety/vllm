# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

logger = logging.getLogger(__name__)


def _resolve_int_env(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _get_num_experts_per_tok(hf_config) -> int:
    """Resolve the per-token expert count from the HF config.

    Different model families store this under different attribute names
    (e.g. ``num_experts_per_tok`` for DeepSeek, ``top_k_experts`` for Gemma 4).
    """
    val = getattr(hf_config, "num_experts_per_tok", None)
    if val is None:
        val = getattr(hf_config, "top_k_experts", None)
    if val is None:
        raise ValueError(
            "Cannot determine num_experts_per_tok: HF config has neither "
            "'num_experts_per_tok' nor 'top_k_experts'"
        )
    return val


def get_num_experts(hf_config) -> int:
    """Resolve ``num_experts`` across HuggingFace config naming conventions.

    Different MoE model families expose this under different keys:
      - ``num_experts``: Mixtral, Qwen2-MoE, Qwen3-MoE
      - ``n_routed_experts``: DeepSeek-V2/V3
      - ``num_local_experts``: Mixtral (older exports)
    """
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(hf_config, key, None)
        if val is not None:
            return val
    raise ValueError(
        "Could not resolve num_experts from model config. "
        "Expected one of 'num_experts', 'n_routed_experts', "
        "or 'num_local_experts'."
    )


class RoutedExpertsCapturer:
    """Worker-side capturer for routed experts, lives on GPU.

    Layer-level hooks call :meth:`capture` from inside the forward pass
    with the per-layer ``topk_ids`` tensor. The tensor is sliced to the
    tokens owned by this DP rank and written into a preallocated device
    buffer. At the end of the step, :class:`GPUModelRunner` reads the
    device buffer, issues a D2H copy into a pinned CPU buffer, and hands
    the result to the scheduler via :class:`RoutedExpertsLists`.

    The device / pinned-CPU transit buffers use ``torch.int32`` (not a
    narrow ``uint8``/``uint16`` sized by ``num_experts``). This keeps the
    SP all-gather path free of dtype casts, matches the router's native
    ``topk_ids`` indices dtype more closely, and costs only a few MB per
    worker (``max_num_batched_tokens * num_layers * top_k * 4`` bytes).
    The scheduler-side slot buffer
    (``RoutedExpertsManager.routed_experts_by_slot``) still uses the
    narrow dtype -- numpy fancy-index assignment in ``store_batch``
    narrows the data on the way in.

    Invariants:
        - One instance per worker; shape is fixed at init and covers the
          worst-case step (``max_num_batched_tokens`` tokens).
        - :meth:`clear_buffer` is called at the start of every step, so
          unused slots stay zero.
        - ``device_buffer.dtype`` is ``torch.int32``.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        num_layers = hf_config.num_hidden_layers
        logger.info(
            "RoutedExpertsCapturer: allocating buffer with "
            "max_tokens=%d, num_layers=%d, num_experts_per_tok=%d "
            "(hf_config.model_type=%s)",
            max_num_batched_tokens,
            num_layers,
            num_experts_per_tok,
            getattr(hf_config, "model_type", "unknown"),
        )
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                num_layers,
                num_experts_per_tok,
            ),
            # Use int32 for the device / host transit buffers: it
            # matches the router's native topk_ids dtype, is universally
            # supported by NCCL (uint8/uint16 are version-dependent),
            # and the extra bytes are small (few MB per worker). The
            # big scheduler-side slot buffer stays narrow.
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Capture expert routing decisions for a specific layer.

        Under data parallelism, ``topk_ids`` may have three different batch
        layouts depending on where the DP combine happens and whether
        Sequence Parallelism (SP) is active for the MoE layer:
          - ``n == total`` (naive dispatch): all DP ranks' tokens are
            concatenated before routing; we slice out this rank's span
            using the cumulative per-rank counts.
          - ``n == token_num_per_dp`` (modular-kernel path): DP combine
            happens inside ``quant_method.apply``; ``select_experts`` only
            ever sees this rank's tokens, so we take the whole tensor.
          - ``n == ceil(token_num_per_dp / tp_size)`` (SP + modular-kernel
            path): tokens were split along dim=0 across the TP group by
            ``_sequence_parallel_context``
            (``moe_runner_base.py:_sequence_parallel_context``), so each
            TP rank only sees its shard. We all-gather along dim=0 to
            reconstruct this DP rank's full routing tensor. SP pads with
            ceil-div (see ``_compute_sp_num_tokens`` in
            ``forward_context.py``), so the gathered tensor may contain a
            few trailing padding rows which are trimmed by the downstream
            ``[:token_num_per_dp]`` slice.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]

            if n == total:
                # Naive dispatch: all DP ranks' tokens concatenated
                # before routing. This rank owns tokens
                # [end_loc - token_num_per_dp, end_loc).
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif n == token_num_per_dp:
                # Modular-kernel path: DP combine happens inside
                # quant_method.apply; select_experts only sees this
                # rank's tokens, take the whole tensor.
                start_loc = 0
                end_loc = token_num_per_dp
            elif (
                self.tp_size > 1
                and n != token_num_per_dp
                and n == (token_num_per_dp + self.tp_size - 1) // self.tp_size
            ):
                # SP + modular-kernel path. All-gather across the TP
                # group along dim=0 to reconstruct the full per-DP-rank
                # tensor; keep only the first ``token_num_per_dp`` rows
                # (trailing rows are SP ceil-div padding). The TP group
                # is always initialized on real rollout workers, and
                # every rank in the group reaches this branch in
                # lockstep (bind is per-FusedMoEFactory layer, SP is a global
                # condition), so a bare all_gather here will not
                # deadlock -- let it raise if the precondition is
                # violated rather than skip silently.
                #
                # ``topk_ids`` is already whatever the router produced
                # (typically int32/int64, both supported by NCCL); the
                # downstream ``device_buffer[...] = topk_ids[...]``
                # setitem narrows into int32 automatically.
                topk_ids = get_tp_group().all_gather(topk_ids, dim=0)
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                sp_expected = (
                    (token_num_per_dp + self.tp_size - 1) // self.tp_size
                    if self.tp_size > 0
                    else -1
                )
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch "
                    f"dim {n} (expected {total}, {token_num_per_dp}, "
                    f"or {sp_expected} for dp_rank={self.dp_rank}, "
                    f"tp_size={self.tp_size})"
                )

        # Defensive: model may expose more layers than the capture buffer
        # was sized for (unusual, but guards against miss-config).
        if layer_id >= self.device_buffer.shape[1]:
            return

        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def clear_buffer(self) -> None:
        """Zero the device buffer. Called at the start of every step so
        slots belonging to finished / preempted tokens don't leak into
        the next step.
        """
        self.device_buffer.zero_()

    def get_device_buffer(self) -> torch.Tensor:
        """Return the underlying device buffer so the model runner can
        issue the D2H copy. The tensor is shared; callers must either
        clone or fully drain it before the next forward pass runs
        :meth:`clear_buffer`.
        """
        return self.device_buffer


class RouterTopKBitmapDumper:
    """Append router top-k decisions as per-token expert bitmaps.

    This diagnostic dumper is intentionally independent of EPLB. It records
    logical expert IDs immediately after router top-k selection and before any
    EPLB remapping. Each worker writes rank-specific files to avoid multi-rank
    append races on shared filesystems.
    """

    def __init__(self, output_dir: str, vllm_config: VllmConfig) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.rank = _resolve_int_env("RANK", "SLURM_PROCID") or 0
        self.local_rank = _resolve_int_env("LOCAL_RANK", "SLURM_LOCALID")
        self.role = os.environ.get("VLLM_ROUTER_TOPK_BITMAP_ROLE")
        self._lock = threading.Lock()
        self._step_by_layer: dict[int, int] = defaultdict(int)
        self._engine_iteration: int | None = None
        self._request_spans: list[dict[str, object]] = []
        logger.info(
            "RouterTopKBitmapDumper: writing router top-k bitmaps to %s "
            "(rank=%d, local_rank=%s, role=%s)",
            self.output_dir,
            self.rank,
            self.local_rank,
            self.role,
        )

    def set_engine_iteration(self, engine_iteration: int) -> None:
        self._engine_iteration = engine_iteration

    def set_request_spans(
        self,
        req_ids: Sequence[str],
        token_counts: Sequence[int],
        request_headers: Mapping[str, Mapping[str, str] | None],
    ) -> None:
        """Set request/token spans for the next model execution.

        Router hooks only receive the flattened token-by-token ``topk_ids``
        tensor. The runner owns the batch ordering and therefore supplies the
        request spans separately so the dumper can preserve request metadata
        without trying to infer it from expert IDs.
        """
        if len(req_ids) != len(token_counts):
            raise ValueError(
                "RouterTopKBitmapDumper: req_ids and token_counts must have "
                "the same length"
            )

        spans: list[dict[str, object]] = []
        token_start = 0
        for req_id, raw_token_count in zip(req_ids, token_counts, strict=True):
            token_count = int(raw_token_count)
            if token_count < 0:
                raise ValueError(
                    "RouterTopKBitmapDumper: token count cannot be negative"
                )
            headers = request_headers.get(req_id) or {}
            spans.append(
                {
                    "request_id": headers.get("x-request-id", req_id),
                    "internal_request_id": req_id,
                    "correlation_id": headers.get("x-correlation-id"),
                    "token_start": token_start,
                    "token_count": token_count,
                }
            )
            token_start += token_count
        self._request_spans = spans

    def _slice_request_spans(self, start: int, end: int) -> list[dict[str, object]]:
        """Clip request spans to the rows written by this worker."""
        if not self._request_spans:
            return []

        source_tokens = sum(
            int(span["token_count"]) for span in self._request_spans
        )
        captured_tokens = end - start
        if start < 0 or end < start:
            raise AssertionError(
                "RouterTopKBitmapDumper: top-k token span exceeds the "
                f"request metadata span (start={start}, end={end}, "
                f"metadata_tokens={source_tokens})"
            )

        # In internal DP dispatch, the router can see the concatenated token
        # tensor while the runner only knows this rank's request list. In
        # that case the metadata is local to the captured slice even though
        # ``start`` is a global DP offset.
        if source_tokens == captured_tokens:
            start, end = 0, source_tokens
        elif end > source_tokens:
            raise AssertionError(
                "RouterTopKBitmapDumper: top-k token span exceeds the "
                f"request metadata span (start={start}, end={end}, "
                f"metadata_tokens={source_tokens})"
            )

        clipped: list[dict[str, object]] = []
        for span in self._request_spans:
            span_start = int(span["token_start"])
            span_end = span_start + int(span["token_count"])
            overlap_start = max(start, span_start)
            overlap_end = min(end, span_end)
            if overlap_start >= overlap_end:
                continue
            clipped_span = dict(span)
            clipped_span["token_start"] = overlap_start - start
            clipped_span["token_count"] = overlap_end - overlap_start
            clipped.append(clipped_span)

        clipped_tokens = sum(int(span["token_count"]) for span in clipped)
        if clipped_tokens != end - start:
            raise AssertionError(
                "RouterTopKBitmapDumper: request metadata does not cover "
                f"the captured token rows (covered={clipped_tokens}, "
                f"captured={end - start})"
            )
        return clipped

    def _slice_topk_ids_with_bounds(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        ctx = get_forward_context()
        if ctx.dp_metadata is None:
            return topk_ids, 0, topk_ids.shape[0]

        num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
        token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
        total = int(num_tokens_dp.sum().item())
        n = topk_ids.shape[0]

        if n == total:
            cumsum = torch.cumsum(num_tokens_dp, dim=0)
            end_loc = int(cumsum[self.dp_rank].item())
            start_loc = end_loc - token_num_per_dp
            return topk_ids[start_loc:end_loc, :], start_loc, end_loc
        if n == token_num_per_dp:
            return topk_ids, 0, n
        if (
            self.tp_size > 1
            and n != token_num_per_dp
            and n == (token_num_per_dp + self.tp_size - 1) // self.tp_size
        ):
            gathered = get_tp_group().all_gather(topk_ids, dim=0)
            return gathered[:token_num_per_dp, :], 0, token_num_per_dp

        sp_expected = (
            (token_num_per_dp + self.tp_size - 1) // self.tp_size
            if self.tp_size > 0
            else -1
        )
        raise AssertionError(
            "RouterTopKBitmapDumper: unexpected topk_ids batch dim "
            f"{n} (expected {total}, {token_num_per_dp}, or {sp_expected} "
            f"for dp_rank={self.dp_rank}, tp_size={self.tp_size})"
        )

    def _slice_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        return self._slice_topk_ids_with_bounds(topk_ids)[0]

    @staticmethod
    def _encode_bitmap_bytes(
        topk_ids_cpu: torch.Tensor,
        num_logical_experts: int,
    ) -> bytes:
        words_per_token = (num_logical_experts + 63) // 64
        payload = bytearray(topk_ids_cpu.shape[0] * words_per_token * 8)
        offset = 0
        for row in topk_ids_cpu.tolist():
            words = [0] * words_per_token
            for expert_id in row:
                expert = int(expert_id)
                if expert < 0 or expert >= num_logical_experts:
                    raise ValueError(
                        "RouterTopKBitmapDumper: expert id "
                        f"{expert} outside [0, {num_logical_experts})"
                    )
                words[expert // 64] |= 1 << (expert % 64)
            for word in words:
                struct.pack_into("<Q", payload, offset, word)
                offset += 8
        return bytes(payload)

    @staticmethod
    def _counts(
        topk_ids_cpu: torch.Tensor, num_logical_experts: int
    ) -> list[list[int]]:
        flat = topk_ids_cpu.reshape(-1)
        if flat.numel() == 0:
            return []
        counts = torch.bincount(flat, minlength=num_logical_experts)
        nz = torch.nonzero(counts, as_tuple=False).flatten().tolist()
        return [[int(i), int(counts[i].item())] for i in nz]

    def capture(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        num_logical_experts: int,
    ) -> None:
        if topk_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "RouterTopKBitmapDumper cannot capture router top-k IDs during "
                "CUDA graph capture; run the metrics path in eager mode"
            )

        sliced, start_loc, end_loc = self._slice_topk_ids_with_bounds(topk_ids)
        topk_ids_cpu = sliced.detach().to("cpu", dtype=torch.int64)
        num_tokens = int(topk_ids_cpu.shape[0])
        top_k = int(topk_ids_cpu.shape[1]) if topk_ids_cpu.ndim == 2 else 0
        words_per_token = (num_logical_experts + 63) // 64
        payload = self._encode_bitmap_bytes(topk_ids_cpu, num_logical_experts)
        counts = self._counts(topk_ids_cpu, num_logical_experts)
        request_spans = self._slice_request_spans(start_loc, end_loc)
        if request_spans and sum(
            int(span["token_count"]) for span in request_spans
        ) != num_tokens:
            raise AssertionError(
                "RouterTopKBitmapDumper: request metadata token count does not "
                "match captured top-k rows"
            )

        base = f"layer_{layer_id:03d}.rank_{self.rank:03d}"
        bitmap_path = self.output_dir / f"{base}.token_expert_bitmap.u64"
        jsonl_path = self.output_dir / f"{base}.jsonl"

        with self._lock:
            record_sequence = self._step_by_layer[layer_id]
            self._step_by_layer[layer_id] += 1
            engine_iteration = self._engine_iteration
            step = engine_iteration if engine_iteration is not None else record_sequence
            offset_bytes = bitmap_path.stat().st_size if bitmap_path.exists() else 0
            with bitmap_path.open("ab") as f:
                f.write(payload)
            record = {
                "rank": self.rank,
                "local_rank": self.local_rank,
                "role": self.role,
                "step": step,
                "engine_iteration": engine_iteration,
                "record_sequence": record_sequence,
                "timestamp_ns": time.time_ns(),
                "layer": layer_id,
                "num_logical_experts": num_logical_experts,
                "num_tokens": num_tokens,
                "top_k": top_k,
                "counts": counts,
                "request_spans": request_spans,
                "token_expert_bitmap": {
                    "path": bitmap_path.name,
                    "offset_bytes": offset_bytes,
                    "num_tokens": num_tokens,
                    "words_per_token": words_per_token,
                    "dtype": "uint64_le",
                },
            }
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")


class RoutedExpertsManager:
    """Scheduler-side slot-indexed buffer for routed experts.

    Lives on CPU in the scheduler process. Each slot corresponds to
    ``block_id * block_size + offset_in_block`` where ``block_id`` is
    drawn from the physical KV-cache block pool, so routing data is
    tied to physical blocks and naturally survives preemption for
    prefix-cached blocks (prefix hits re-expose the same slots).

    Data flow per step:
      1. Worker D2Hs its device capture buffer into
         :class:`RoutedExpertsLists` and returns it via
         :class:`ModelRunnerOutput`.
      2. Scheduler calls :meth:`store_batch` with that step's
         ``(routing_data, slot_mapping)`` — a single CPU->CPU
         fancy-index assign, ~few MB per step.
      3. On request completion / abort / preemption, the scheduler
         calls :meth:`get` with the request's block IDs to recover
         the full per-token routing.

    Memory: ``routed_experts_by_slot`` is sized for the whole block
    pool (``num_blocks * block_size`` slots). For large block pools
    this can reach multiple GB; see the init log for the exact size.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Pick the attention group for block/slot mapping. We require
        # a FullAttentionSpec group rather than any AttentionSpec to
        # stay consistent with the worker-side lookup in
        # ``GPUModelRunner._get_attention_kv_cache_gid``; hybrid models
        # (Mamba / linear attention) also have other AttentionSpec
        # groups whose slot layout differs.
        self.attn_gid = next(
            gid
            for gid, g in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(g.kv_cache_spec, FullAttentionSpec)
        )
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

        # All kv_cache_groups share the same physical block pool, so
        # block IDs span [0, num_blocks) regardless of how many groups
        # exist. Sizing to the full pool avoids index-out-of-range
        # when different groups happen to land on the same block.
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        # Expert IDs are 0..num_experts-1; uint8 fits 256 distinct
        # values so the boundary is ``<= 256`` (NOT ``< 256``). Keeping
        # this narrow matters because the slot buffer is sized for the
        # whole block pool and can reach multiple GB.
        expert_id_dtype = np.uint8 if num_experts <= 256 else np.uint16
        self.routed_experts_by_slot = np.zeros(
            (
                max_num_slots,
                hf_config.num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=expert_id_dtype,
        )
        logger.info(
            "RoutedExpertsManager CPU buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s)",
            self.routed_experts_by_slot.nbytes / 1e9,
            max_num_slots,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
        )

    def store_batch(self, data: np.ndarray, slot_mapping: np.ndarray) -> None:
        """Persist one step's routed experts into the slot buffer.

        Equivalent to ``slot_buffer[slot_mapping] = data``; numpy fancy
        indexing handles repeated / out-of-order indices. Called once
        per scheduler step in ``update_from_output``.
        """
        self.routed_experts_by_slot[slot_mapping] = data

    def get(
        self,
        block_ids: list[int],
        num_tokens: int,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read routed experts data for a completed / preempted request.

        Reconstructs a per-token slot_mapping from the request's block
        IDs and returns the routing slice. Because numpy fancy indexing
        returns a **copy** (not a view), the returned ndarray is safe
        to hold across subsequent :meth:`store_batch` calls — do not
        replace the fancy index with a slice without re-verifying.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of tokens that have gone through a forward
                pass and therefore have routing data written to their
                slots (typically ``request.num_tokens - 1``; the last
                sampled token has not been forwarded yet). Slots beyond
                ``request.num_computed_tokens`` are zero-initialized.
            token_start: Skip the first ``token_start`` tokens from the
                result. The slot_mapping is sliced before the fancy-index
                read, so only the requested slots are fetched — no large
                intermediate array is allocated. Clamped to
                ``[0, num_tokens]`` automatically.

        Returns:
            Array of shape (num_tokens - token_start, num_layers,
            num_experts_per_tok).
        """
        bs = self.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(bs)
        # slot = block_id * block_size + offset_in_block; flatten the
        # (num_blocks, block_size) grid and trim to num_tokens, then
        # skip the first token_start entries so only the requested
        # range is fetched in a single fancy-index read.
        slot_mapping = (
            block_ids_array.reshape(-1, 1) * bs + block_offsets.reshape(1, -1)
        ).flatten()[:num_tokens]
        slot_mapping = slot_mapping[token_start:]
        return self.routed_experts_by_slot[slot_mapping]
