from .config import BiatronConfig
from .modeling_biatron import BiatronForCausalLM, BiatronModel

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("Biatron", BiatronConfig)
AutoModelForCausalLM.register(BiatronConfig, BiatronForCausalLM)
