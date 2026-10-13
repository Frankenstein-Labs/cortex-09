"""Configuration for Cortex.09.

Cortex.09 keeps the DeepSeek-V4-Pro architecture. This config subclasses the
upstream implementation and only changes the `model_type` identifier so the
model can be loaded through the transformers Auto-classes under the name
`cortex09`.
"""

from transformers.models.deepseek_v4 import DeepseekV4Config


class Cortex09Config(DeepseekV4Config):
    model_type = "cortex09"
