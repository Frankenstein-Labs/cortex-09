"""Cortex.09 - language model architecture.

Importing this package registers the `cortex09` model type with the
transformers Auto-classes, so a Cortex.09 checkpoint can be loaded with
`AutoConfig` / `AutoModelForCausalLM` once its code is trusted.
"""

from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from .configuration_cortex09 import Cortex09Config
from .modeling_cortex09 import Cortex09ForCausalLM

# Make transformers aware of the new model_type -> class mapping.
AutoConfig.register(Cortex09Config.model_type, Cortex09Config, exist_ok=True)
MODEL_FOR_CAUSAL_LM_MAPPING.register(Cortex09Config, Cortex09ForCausalLM, exist_ok=True)

__all__ = ["Cortex09Config", "Cortex09ForCausalLM"]
