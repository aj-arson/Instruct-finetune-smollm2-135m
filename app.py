import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from inferance import load_model, load_tokenizer, generate

@st.cache_resource
def load_resources():
    base_model = load_model(False)
    finetuned_model = load_model(True)
    tokenizer = load_tokenizer()
    device = base_model.device
    return base_model, finetuned_model, tokenizer, device

base_model, finetuned_model, tokenizer, device = load_resources()

if "base_model_reponse" not in st.session_state:
    st.session_state.base_model_reponse = None

if "finetuned_model_reponse" not in st.session_state:
    st.session_state.finetuned_model_reponse = None

if "max_new_tokens" not in st.session_state:
    st.session_state.max_new_tokens = None

if "finetune_gen" not in st.session_state:
    st.session_state.finetune_gen = None

if "base_gen" not in st.session_state:
    st.session_state.base_gen = None

if "gen" not in st.session_state:
     st.session_state.gen = None

st.title("SFT-SmolLM2-135M-Instruct")

st.session_state.max_new_tokens = st.slider(label="Max new tokens", min_value=20, max_value=400, step=1, value=50)
st.caption("Note: This is not a huge model, so don't have high expectations.")

with st.container(border=True, height="content", width="stretch"):
    instruction = st.text_area(label='please input your instruction here:', max_chars=2000, placeholder="What's on your mind?")  

    if st.button(label='Generate', use_container_width=True) and instruction.strip():
        st.session_state.finetune_gen = True  
        st.session_state.base_gen = True
        st.session_state.gen = True

if st.session_state.finetune_gen and st.session_state.finetune_gen:
     with st.spinner(text="Model analyzing input..."):
          with ThreadPoolExecutor(max_workers = 2) as pool:
               finetuned_model_future = pool.submit(generate, finetuned_model, tokenizer, instruction, return_only_answer=True, max_new_tokens=st.session_state.max_new_tokens)
               base_model_future = pool.submit(generate, base_model, tokenizer, instruction, return_only_answer=True, max_new_tokens=st.session_state.max_new_tokens)

if st.session_state.gen:
    with st.container(border=True, height="content", width="stretch"):
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.caption("Finetuned Model")
            if st.session_state.finetune_gen:
                with st.spinner(text="Model analyzing input..."):
                    # st.session_state.finetuned_model_reponse = generate(finetuned_model, tokenizer, instruction, return_only_answer=True, max_new_tokens=st.session_state.max_new_tokens)
                    st.session_state.finetuned_model_reponse = finetuned_model_future.result()
                    st.session_state.finetune_gen = None

                if st.session_state.finetuned_model_reponse:
                        with st.container(border=True, height="stretch", width="stretch"):
                            st.write(st.session_state.finetuned_model_reponse)

        with col2:
                st.caption("Base Model")
                if st.session_state.base_gen:
                    with st.spinner(text="Model analyzing input..."):
                        # st.session_state.base_model_reponse = generate(base_model, tokenizer, instruction, return_only_answer=True, max_new_tokens=st.session_state.max_new_tokens)
                        st.session_state.base_model_reponse = base_model_future.result()
                        st.session_state.base_gen = None
        
                    if st.session_state.base_model_reponse:
                            with st.container(border=True, height="stretch", width="stretch"):
                                st.write(st.session_state.base_model_reponse)
    st.session_state.gen = None