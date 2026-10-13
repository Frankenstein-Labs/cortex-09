"""Modeling code for Cortex.09.

Cortex.09 keeps the DeepSeek-V4-Pro architecture unchanged. The model class
below subclasses the upstream implementation, exposing it under the Cortex.09
name and binding it to :class:`Cortex09Config`.
"""

from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

from .configuration_cortex09 import Cortex09Config


class Cortex09ForCausalLM(DeepseekV4ForCausalLM):
    config_class = Cortex09Config
