"""Inspect what's at the query position in π0's prefix.

If position 788 (last_valid_pos) is a special token (EOS, BOS, padding,
SEP), attention from it is uninformative regardless of the model's
visual understanding. The right query is the last MEANINGFUL CONTENT
token.
"""
from __future__ import annotations

import numpy as np
import torch

from emboviz.datasets.lerobot_libero import PhysicalIntelligenceLiberoSource
from emboviz.models.pi0 import Pi0Adapter

scene = PhysicalIntelligenceLiberoSource().load_trajectory(0).frames[0]
adapter = Pi0Adapter(config_name="pi0_libero", use_pytorch=True)
pi0_model = adapter._policy._model
paligemma = pi0_model.paligemma_with_expert.paligemma

# Capture lang_tokens + prefix_pad_masks via patching
captured = {}
original_embed = pi0_model.embed_prefix
def patched_embed(images, img_masks, lang_tokens, lang_masks):
    result = original_embed(images, img_masks, lang_tokens, lang_masks)
    captured["lang_tokens"]      = lang_tokens.detach().cpu()
    captured["lang_masks"]       = lang_masks.detach().cpu()
    captured["prefix_pad_masks"] = result[1].detach().cpu()
    captured["prefix_embs_shape"] = result[0].shape
    return result
pi0_model.embed_prefix = patched_embed
try:
    obs = adapter._observation_builder(scene)
    with torch.inference_mode():
        _ = adapter._policy.infer(obs)
finally:
    pi0_model.embed_prefix = original_embed

# Inspect
lang_tokens = captured["lang_tokens"][0].tolist()
lang_masks  = captured["lang_masks"][0].tolist()
pad_masks   = captured["prefix_pad_masks"][0].tolist()
prefix_seq_len = captured["prefix_embs_shape"][1]

# n_image_tokens total: from where lang_tokens start in prefix
# (= prefix_seq_len - len(lang_tokens) when packed without padding logic mixing)
n_lang = len(lang_tokens)
n_img  = prefix_seq_len - n_lang
print(f"prefix_seq_len = {prefix_seq_len}, n_img = {n_img}, n_lang = {n_lang}")
print()

print(f"lang_tokens (raw ids): {lang_tokens}")
print(f"lang_masks (True=real text): {lang_masks}")
print()

# Decode each lang token using openpi's own tokenizer (no HF gated repo).
from openpi.models import tokenizer as _opi_tok
opi_tokenizer = _opi_tok.PaligemmaTokenizer()
# PaligemmaTokenizer wraps a sentencepiece processor; access it directly.
sp = opi_tokenizer._tokenizer
decoded = [sp.id_to_piece(int(tid)) for tid in lang_tokens]
print(f"decoded lang tokens (per-token):")
for i, (tid, dec, mask, pad) in enumerate(zip(lang_tokens, decoded, lang_masks, pad_masks[n_img:])):
    flag = " ←REAL" if mask else "       "
    print(f"  lang_pos={i:3d}  prefix_pos={n_img+i:3d}  id={tid:6d}  mask={int(mask)}  pad_in_prefix={int(pad)}  {flag}  {dec!r}")

# What is at the last_valid_pos?
last_valid = max(i for i, p in enumerate(pad_masks) if p)
print(f"\nlast_valid_pos in prefix = {last_valid}")
print(f"This is lang position {last_valid - n_img}, lang_tokens[{last_valid - n_img}] = {lang_tokens[last_valid - n_img]} = {decoded[last_valid - n_img]!r}")

# What's the FULL decoded text?
real_token_ids = [int(tid) for tid, m in zip(lang_tokens, lang_masks) if m]
print(f"\nFull decoded text (real tokens only): {sp.decode_ids(real_token_ids)!r}")
