# In this file, we define load_model
# It runs once at server startup to load the model to a GPU

from transformers import GPTJForCausalLM
import torch

def load_model():
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.cuda()
    return model