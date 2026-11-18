import huggingface_hub
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

huggingface_hub.login(token=os.getenv("HF_TOKEN"))
saved_path = "saved_checkpoint"
model = AutoModelForCausalLM.from_pretrained(saved_path)
tokenizer = AutoTokenizer.from_pretrained(saved_path)


model.push_to_hub("AJ700/instruct_finetuned_smollm_v1")
tokenizer.push_to_hub("AJ700/instruct_finetuned_smollm_v1")