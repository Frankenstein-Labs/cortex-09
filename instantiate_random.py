"""Instantiate Cortex09ForCausalLM with random weights (no checkpoint).

Two modes:
  python instantiate_random.py mini    -> tiny model, runs on CPU (validated)
  python instantiate_random.py full    -> full 1.57T architecture, meta device (no RAM)

For the `full` mode, run this script from the folder that holds config.json.
"""

import sys

import torch
from transformers import AutoConfig

import cortex09  # registers model_type "cortex09" with the transformers Auto-classes
from cortex09 import Cortex09ForCausalLM

mode = sys.argv[1] if len(sys.argv) > 1 else "mini"


def count(model):
    return sum(p.numel() for p in model.parameters())


if mode == "mini":
    cfg = AutoConfig.for_model(
        "cortex09",
        vocab_size=1024,
        hidden_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=8,
        head_dim=64,
        q_lora_rank=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=16,
        max_position_embeddings=512,
        sliding_window=32,
    )
    model = Cortex09ForCausalLM(cfg)
    model.eval()
    print("params:", count(model))
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    print("loss:", float(out.loss), "logits:", tuple(out.logits.shape))

elif mode == "full":
    # Random weights, but never materialized: parameters live on the meta device.
    cfg = AutoConfig.from_pretrained(".")
    with torch.device("meta"):
        model = Cortex09ForCausalLM(cfg)
    n = count(model)
    print(f"full architecture params: {n} ({n / 1e12:.3f} T)")
else:
    raise SystemExit("mode must be 'mini' or 'full'")
