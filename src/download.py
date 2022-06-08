# In this file, we define download_model
# It runs during container build time to get model weights locally

from transformers import GPTJForCausalLM
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

if __name__ == "__main__":
    download_model()