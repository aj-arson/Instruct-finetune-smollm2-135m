from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SYSTEM_PROMPT = """You are an helpful assistant. 
Your name is 'AARSON V1'. You follow user instructions clearly. 
You will use your knowledge and the provided context i.e., if there is any from the user, to help the user as much as you can."""

MAX_LEN = 1024
chat_template = """{%- for message in messages -%}
    {%- if message['role']=='user' -%}
        {{- '<|im_start|>'+'User: '+message['content']+'<|im_end|>' -}}
    {%- elif message['role']=='assistant' -%}
        {{- '<|im_start|>'+'Assistant: '+message['content']+'<|im_end|>' -}}
    {%- elif message['role']=='system' -%}
            {{- '<|im_start|>'+'System: '+message['content']+'<|im_end|>' -}}
    {%- endif -%}
    {%- endfor -%}"""

def load_model(load_checkpoint=False):
    if load_checkpoint:
        model = AutoModelForCausalLM.from_pretrained("saved_checkpoint_final", cache_dir=r"model").to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", cache_dir=r"model").to(device)
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", cache_dir=r"tokenizer")
    tokenizer.chat_template = chat_template
    return tokenizer

def build_chat_messages(tokenizer, instruction, tokenize=True):
    tail = "<|im_start|>Assistant:"
    messages = {"messages":[{'role':'system', 'content':SYSTEM_PROMPT},
               {'role':'user','content':instruction}]}
    if tokenize:
        tokenized = tokenizer.apply_chat_template(messages['messages'], tokenize=tokenize, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        tokenized_tail = tokenizer(tail, return_tensors="pt")
        
        return {'input_ids':torch.cat((tokenized['input_ids'], tokenized_tail['input_ids']), dim=-1),
                'attention_mask':torch.cat((tokenized['attention_mask'], tokenized_tail['attention_mask']), dim=-1)}
    else:
        return tokenizer.apply_chat_template(messages['messages'], tokenize=tokenize, truncation=True, max_length=MAX_LEN, return_tensors="pt")+tail

def generate(model, tokenizer, instruction, return_only_answer=False, max_new_tokens=None):
    chat_messages = build_chat_messages(tokenizer, instruction, True)
    input_ids = chat_messages['input_ids'].to(device)
    if max_new_tokens is not None:
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, stop_strings=["<|im_end|>"],   # a list is the documented form
    tokenizer=tokenizer).squeeze()
    else:  
        output_ids = model.generate(input_ids, stop_strings="<|im_end|>").squeeze()
    if return_only_answer:
        output_ids = output_ids[input_ids.shape[-1]:]
    return tokenizer.decode(output_ids)