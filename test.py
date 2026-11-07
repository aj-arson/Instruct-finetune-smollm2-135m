from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_file_path = 'eval_prompts.csv'

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
MAX_NEW_TOKENS = 1000

def generate(prompt, max_new_tokens=256):
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


if __name__=="__main__":
    data = pd.read_csv(input_file_path)
    data['generated_response'] = None

    print(f"Running on {model.device}")

    for i in tqdm(range(len(data))):
        instruction = data.loc[i, 'prompt']
        answer = generate(instruction, max_new_tokens=MAX_NEW_TOKENS)
        data.loc[i, 'generated_response'] = answer
    data.to_csv('huggingface_released_sft_model.csv')