import torch
from llavajp.model.llava_llama import LlavaLlamaForCausalLM
from llavajp.model.llava_gpt2 import LlavaGpt2ForCausalLM

# load model
kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
model = LlavaLlamaForCausalLM.from_pretrained(
    "team-hatakeyama-phase2/Tanuki-8B-vision-v4-checkpoint-26000", low_cpu_mem_usage=True, cache_dir="./cache"
)

#model = LlavaGpt2ForCausalLM.from_pretrained(
#    "hibikaze/finetune-llava-v1.5-japanese-gpt2-small_test-checkpoint-1200", low_cpu_mem_usage=True, cache_dir="./cache", attn_implementation="eager",
#)

# load vision tower
model.get_vision_tower().load_model()

# Save state dict
torch.save(model.state_dict(), "convert_hf/model_state_dict.bin")
