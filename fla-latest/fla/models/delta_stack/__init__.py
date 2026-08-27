
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.delta_stack.configuration_delta_stack import DeltaStackConfig
from fla.models.delta_stack.modeling_delta_stack import DeltaStackForCausalLM, DeltaStackModel

AutoConfig.register(DeltaStackConfig.model_type, DeltaStackConfig, exist_ok=True)
AutoModel.register(DeltaStackConfig, DeltaStackModel, exist_ok=True)
AutoModelForCausalLM.register(DeltaStackConfig, DeltaStackForCausalLM, exist_ok=True)

__all__ = ['DeltaStackConfig', 'DeltaStackForCausalLM', 'DeltaStackModel']
