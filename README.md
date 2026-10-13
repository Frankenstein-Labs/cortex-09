---
license: mit
library_name: transformers
---
# Cortex.09

Cortex.09 is a large-scale Mixture-of-Experts language model **architecture**. This
repository ships the **code and configuration only** — no pretrained weights. It is
meant for instantiating the architecture with randomly initialized weights, inspecting
it, and training it from scratch at a scale you choose.

The architecture is derived from **DeepSeek-V4-Pro-0813** (MIT licensed). Only naming
was changed for the Cortex.09 brand; the model structure itself is unchanged. See the
[License and attribution](#license-and-attribution) section.

## Architecture

Cortex.09 keeps the full DeepSeek-V4-Pro structure:

- **Mixture-of-Experts (MoE)** feed-forward layers, with routed experts plus one shared
  expert, and a **Hash-MoE** bootstrap on the first layers.
- **Multi-head Latent Attention (MLA)** with low-rank Q/O projections and grouped output.
- **Manifold-Constrained Hyper-Connections (mHC)** residual mixing.
- **Sliding-window attention** with **compressed sparse attention** (CSA) and
  **heavily compressed attention** (HCA) layers, plus a **Lightning Indexer**.
- **DSpark** speculative decoding module.

The reference implementation lives in [`inference/model.py`](inference/model.py). The
same architecture is also available natively through `transformers` (>= 5.0) as the
`cortex09` model type, provided by the [`cortex09/`](cortex09) package in this repo.

Reference full-size configuration: **61 layers, 384 routed experts, ~1.57T parameters.**

## Install

```bash
pip install "transformers>=5.0" "torch>=2.10"
```

The optional `inference/` code (GPU kernels, FP8/FP4, expert parallelism) needs the
extra requirements in [`inference/requirements.txt`](inference/requirements.txt).

## Instantiate with random weights (no checkpoint)

```python
import torch
from transformers import AutoConfig

import cortex09  # registers model_type "cortex09"
from cortex09 import Cortex09ForCausalLM

# Start from the shipped full config, but never materialize the 1.57T parameters:
cfg = AutoConfig.from_pretrained(".")          # reads config.json
with torch.device("meta"):
    model = Cortex09ForCausalLM(cfg)
print(sum(p.numel() for p in model.parameters()))   # 1572997179491

# Or build a small, trainable model that keeps the same architecture:
cfg = AutoConfig.for_model(
    "cortex09", vocab_size=1024, hidden_size=128, moe_intermediate_size=64,
    num_hidden_layers=4, num_attention_heads=8, head_dim=64, q_lora_rank=64,
    n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
    o_groups=2, o_lora_rank=32, index_n_heads=4, index_head_dim=32, index_topk=16,
    max_position_embeddings=512, sliding_window=32,
)
model = Cortex09ForCausalLM(cfg)
```

See [`instantiate_random.py`](instantiate_random.py) (`mini` and `full` modes).

## Tokenizer and chat template

The tokenizer files (`tokenizer.json`, `tokenizer_config.json`) are not versioned here
(they are large, non-code assets). Fetch them separately, without weights:

```bash
hf download deepseek-ai/DeepSeek-V4-Pro-0813 \
  --include "tokenizer.json" --include "tokenizer_config.json" \
  --include "generation_config.json" --local-dir .
```

There is no Jinja chat template. Use the [`encoding/`](encoding/README.md) scripts:

```python
from encoding_cortex09 import encode_messages, parse_message_from_completion_text

messages = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hello!", "reasoning_content": "thinking..."},
    {"role": "user", "content": "1+1=?"},
]
prompt = encode_messages(messages, thinking_mode="thinking", reasoning_effort="max")
```

Note: the chat-control **special tokens** (e.g. `｜DSML｜`) are part of the
tokenizer and were intentionally **not** renamed.

## Inference

The `inference/` code targets the reference GPU implementation. For the original
larger release it is documented in [`inference/README.md`](inference/README.md)
(weight conversion + `torchrun generate.py`). Running the full-size model requires
multi-GPU FP8/FP4 hardware; this repo contains no weights.

## Reduce the model size for a personal machine

Edit `config.json` (or build a config in code) while keeping the same architecture:

| Key | Effect |
| :--- | :--- |
| `num_hidden_layers` | number of blocks |
| `hidden_size` | model width |
| `n_routed_experts` | number of routed experts |
| `num_experts_per_tok` | experts activated per token (compute, not params) |
| `moe_intermediate_size` | expert FFN width |
| `num_attention_heads`, `head_dim`, `q_lora_rank` | attention sizes |
| `o_groups`, `o_lora_rank` | grouped output projection |
| `vocab_size` | embedding and head size |

Constraints: `num_attention_heads` must be divisible by `o_groups`; keep
`num_hash_layers <= num_hidden_layers`; drop `quantization_config` / `expert_dtype`
for BF16 training.

## Training the full model

Training the **full 1.57T** configuration is **not feasible on a personal machine**:
BF16 weights alone are ~3.1TB and optimizer states push this to ~12–15TB, requiring a
large multi-GPU cluster. Realistic personal options are training a reduced configuration
(same architecture, smaller hyperparameters) or parameter-efficient fine-tuning of a
smaller model. Fine-tuning/pretraining the full-size model is out of scope for
individual hardware.

## License and attribution

This repository is licensed under the **MIT License** (see [`LICENSE`](LICENSE)).

The architecture and the reference inference code are derived from
**DeepSeek-V4-Pro-0813** by **DeepSeek-AI**, also MIT licensed:
<https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813>. The original copyright
notice is preserved in `LICENSE`, as required by the MIT License.

Original citation:

```
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```
