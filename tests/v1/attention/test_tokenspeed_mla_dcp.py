# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla import tokenspeed_mla as ts_mla


def _make_impl(
    *,
    dcp_world_size: int,
    dcp_rank: int,
    return_lse: bool,
) -> ts_mla.TokenspeedMLAImpl:
    impl = object.__new__(ts_mla.TokenspeedMLAImpl)
    impl.dcp_world_size = dcp_world_size
    impl.dcp_rank = dcp_rank
    impl.cp_kv_cache_interleave_size = 1
    impl.need_to_return_lse_for_decode = return_lse
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    impl.num_heads = 2
    impl.scale = 1.0
    impl.softmax_scale = 1.0
    impl.output_scale = 1.0
    impl._workspace_buffer = torch.empty(1, dtype=torch.int8)
    return impl


def _metadata(
    *,
    num_decodes: int,
    num_decode_tokens: int,
    dcp_tot_seq_lens: torch.Tensor | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        max_seq_len=128,
        decode=SimpleNamespace(
            block_table=torch.arange(num_decodes * 4, dtype=torch.int32).view(
                num_decodes, 4
            ),
            seq_lens=torch.arange(10, 10 + num_decodes, dtype=torch.int32),
            dcp_tot_seq_lens=dcp_tot_seq_lens,
        ),
    )


def _layer() -> SimpleNamespace:
    return SimpleNamespace(_q_scale_float=2.0, _k_scale_float=3.0)


def test_tokenspeed_dcp_decode_api_detection(monkeypatch: pytest.MonkeyPatch):
    def decode_with_dcp(
        *,
        return_lse=False,
        causal_seqs=None,
        cp_world=1,
        cp_rank=0,
    ):
        return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "tokenspeed_mla",
        SimpleNamespace(tokenspeed_mla_decode=decode_with_dcp),
    )
    assert ts_mla._missing_dcp_decode_kwargs() == set()

    def decode_without_dcp():
        return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "tokenspeed_mla",
        SimpleNamespace(tokenspeed_mla_decode=decode_without_dcp),
    )
    assert ts_mla._missing_dcp_decode_kwargs() == {
        "causal_seqs",
        "cp_rank",
        "cp_world",
        "return_lse",
    }


def test_tokenspeed_dcp_single_token_decode_passes_lse_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        q = kwargs["query"]
        out = torch.empty(q.shape[0], q.shape[1], q.shape[2], 3, dtype=torch.bfloat16)
        lse = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32)
        return out, lse

    monkeypatch.setitem(
        __import__("sys").modules,
        "tokenspeed_mla",
        SimpleNamespace(tokenspeed_mla_decode=fake_decode),
    )

    impl = _make_impl(dcp_world_size=4, dcp_rank=2, return_lse=True)
    metadata = _metadata(
        num_decodes=2,
        num_decode_tokens=2,
        dcp_tot_seq_lens=torch.tensor([33, 44], dtype=torch.int32),
    )
    q = torch.empty(2, 2, 5, dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty(4, 16, 5, dtype=torch.float8_e4m3fn)

    out, lse = impl.forward_mqa(q, kv_cache, metadata, _layer())

    assert out.shape == (2, 2, 3)
    assert lse is not None
    assert lse.shape == (2, 2)
    call = calls.pop()
    assert call["return_lse"] is True
    assert call["cp_world"] == 4
    assert call["cp_rank"] == 2
    torch.testing.assert_close(call["causal_seqs"], metadata.decode.dcp_tot_seq_lens)
    torch.testing.assert_close(call["block_tables"], metadata.decode.block_table)
    torch.testing.assert_close(call["seq_lens"], metadata.decode.seq_lens)


def test_tokenspeed_dcp_mtp_decode_splits_q_and_causal_seqs(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    split_calls = []

    def fake_split_q(**kwargs):
        split_calls.append(kwargs)
        return (
            torch.arange(8, dtype=torch.int32),
            torch.arange(24, dtype=torch.int32).view(8, 3),
        )

    def fake_decode(**kwargs):
        calls.append(kwargs)
        q = kwargs["query"]
        out = torch.empty(q.shape[0], q.shape[1], q.shape[2], 3, dtype=torch.bfloat16)
        lse = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32)
        return out, lse

    monkeypatch.setattr(ts_mla, "dcp_split_q", fake_split_q)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tokenspeed_mla",
        SimpleNamespace(tokenspeed_mla_decode=fake_decode),
    )

    impl = _make_impl(dcp_world_size=4, dcp_rank=1, return_lse=True)
    metadata = _metadata(
        num_decodes=2,
        num_decode_tokens=8,
        dcp_tot_seq_lens=torch.tensor([10, 20], dtype=torch.int32),
    )
    q = torch.empty(8, 2, 5, dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty(4, 16, 5, dtype=torch.float8_e4m3fn)

    out, lse = impl.forward_mqa(q, kv_cache, metadata, _layer())

    assert out.shape == (8, 2, 3)
    assert lse is not None
    assert lse.shape == (8, 2)
    split_call = split_calls.pop()
    assert split_call["num_decodes"] == 2
    assert split_call["tokens_per_req"] == 4
    assert split_call["dcp_world_size"] == 4
    assert split_call["dcp_rank"] == 1

    call = calls.pop()
    assert call["query"].shape == (8, 1, 2, 5)
    torch.testing.assert_close(call["seq_lens"], torch.arange(8, dtype=torch.int32))
    torch.testing.assert_close(
        call["block_tables"], torch.arange(24, dtype=torch.int32).view(8, 3)
    )
    torch.testing.assert_close(
        call["causal_seqs"],
        torch.tensor([7, 8, 9, 10, 17, 18, 19, 20], dtype=torch.int32),
    )
    assert call["return_lse"] is True
    assert call["cp_world"] == 4
    assert call["cp_rank"] == 1
