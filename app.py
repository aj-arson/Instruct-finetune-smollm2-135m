import streamlit as st
from inferance import load_model_and_tokenizer, generate

@st.cache_resource
def load_resources():
    return load_model_and_tokenizer(True)

model, tokenizer, device = load_resources()

if "chat_reponse" not in st.session_state:
    st.session_state.chat_response = "Model response comes here..."

if "max_new_tokens" not in st.session_state:
    st.session_state.chat_response = 200


st.title("SFT-SmolLM2-135M-Instruct")

st.session_state.max_new_tokens = st.slider(label="Max new tokens", min_value=20, max_value=500, step=1, value=200)
st.caption("Note: This is not a huge model, so don't have high expectations.")

with st.container(border=True, height="content", width="stretch"):
    col1, col2 = st.columns([0.5,0.5])
    with col1:
        instruction = st.text_area(label='please input your instruction here:', max_chars=2000, placeholder="What's on your mind?")
        if st.button(label='Generate', use_container_width=True):
            with st.spinner(text="Model analyzing input..."):
                st.session_state.chat_response = generate(model, tokenizer, instruction, return_only_answer=True, max_new_tokens=st.session_state.max_new_tokens)
    with col2:
        with st.container(border=True, height="stretch", width="stretch"):
            st.write(st.session_state.chat_response)
