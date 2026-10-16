# Supported weight schema (AIM CLI v0.1.0)

The model must be a **single-layer single-head autoregressive decoder** with:

- learned token embeddings `E` and low-rank positional embeddings `P_A @ P_B`;
- bias-free RMSNorm with `eps = 1e-8` in pre-attention, pre-FFN and final output positions;
- shared-A query/key/value projection with **tied keys and values**;
- softmax causal attention with scores `dot(q,k)/sqrt(d_model)`;
- bias-free rank-factorized GELU (exact `erf` definition) feedforward block;
- tied token embedding and output readout; deterministic argmax.

Configuration flags: `n_layer=1`, `n_head=1`, `use_rmsnorm=true`, `tie_qkv="shareA_tieKV"`, `pe_type="learned"`, plus integer values for `d_model`, `vocab_size`, `max_seq_len`, `d_ff`, `qkv_rank`, `pos_rank`, `attn_out_rank`, `ffn_rank`. This architecture is not a standard GPT-2 / Llama / HF checkpoint contract.

The expected PyTorch checkpoint has `model_config` and `model_state`. It must use the following tensor names:

| AIM slot | PyTorch `model_state` name | Shape |
|---|---|---|
| `E` | `token_emb.weight` | `[vocab_size, d_model]` |
| `P_A` | `pos_emb.A` | `[max_seq_len, pos_rank]` |
| `P_B` | `pos_emb.B` | `[pos_rank, d_model]` |
| `g1` | `blocks.0.ln1.weight` | `[d_model]` |
| `A` | `blocks.0.attn.qkv_A` | `[d_model, qkv_rank]` |
| `Bq` | `blocks.0.attn.qkv_Bq` | `[qkv_rank, d_model]` |
| `Bkv` | `blocks.0.attn.qkv_Bkv` | `[qkv_rank, d_model]` |
| `O_A` | `blocks.0.attn.proj.A` | `[d_model, attn_out_rank]` |
| `O_B` | `blocks.0.attn.proj.B` | `[attn_out_rank, d_model]` |
| `g2` | `blocks.0.ln2.weight` | `[d_model]` |
| `F1_A` | `blocks.0.mlp.fc1.A` | `[d_model, ffn_rank]` |
| `F1_B` | `blocks.0.mlp.fc1.B` | `[ffn_rank, d_ff]` |
| `F2_A` | `blocks.0.mlp.fc2.A` | `[d_ff, ffn_rank]` |
| `F2_B` | `blocks.0.mlp.fc2.B` | `[ffn_rank, d_model]` |
| `gf` | `ln_f.weight` | `[d_model]` |

The `.pt` bridge also checks `lm_head.weight` exists and equals `token_emb.weight` element-by-element. Non-finite coefficients are rejected.

## Portable JSON

`aim export` emits the following format. It can be used to analyze privately trained weights without installing Python/PyTorch on the analysis machine:

```json
{
  "format": "aim-checkpoint-v0",
  "architecture": "sharedA_tieKV_one_layer_v1",
  "model_config": { "n_layer": 1, "n_head": 1, "use_rmsnorm": true, "tie_qkv": "shareA_tieKV", "pe_type": "learned", "d_model": 4, "vocab_size": 14, "max_seq_len": 33, "d_ff": 6, "qkv_rank": 3, "pos_rank": 2, "attn_out_rank": 3, "ffn_rank": 3 },
  "tensors": {
    "E": { "shape": [14, 4], "data": [0.0] }
  }
}
```

The abbreviated snippet above is **schema illustration only**, not a usable weight file. All slots are mandatory, and `data` must contain every coefficient in row-major order, with the declared shape. In portable JSON the tied output head is a schema assumption (represented by `E` only); it is not checked against a separate output-head tensor. The input's declared architecture is part of the proof premises.

This tool does **not** infer an architecture merely from arbitrary tensor names or accept all generic Hugging Face checkpoints. Unsupported network variants are rejected.
