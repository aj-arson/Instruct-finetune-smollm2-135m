import pandas as pd
from inferance import load_model_and_tokenizer, generate
from tqdm import tqdm

input_file_path = 'eval_prompts.csv'
output_file_path = 'alpaca_dataset_model_output.csv' #or "dolly_dataset_model_output.csv"

data = pd.read_csv(input_file_path)

model, tokenizer, device = load_model_and_tokenizer(load_checkpoint=True)
MAX_NEW_TOKENS = 1000
data['generated_response'] = None


if __name__=='__main__':
    for i in tqdm(range(len(data))):
        instruction = data.loc[i, 'prompt']
        answer = generate(model, tokenizer, instruction, return_only_answer=True, max_new_tokens=MAX_NEW_TOKENS)
        data.loc[i, 'generated_response'] = answer

    data.to_csv(output_file_path)
    print(f'saved to {output_file_path}')