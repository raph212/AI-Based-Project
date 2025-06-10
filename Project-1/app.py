# app.py

import streamlit as st
# Make sure this is the ONLY import statement related to 'summarizer' in this file
from summarizer import TextSummarizer # This line is correct as is

st.set_page_config(page_title="AI Text Summarizer", layout="centered")

@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_summarizer():
    """Loads the TextSummarizer model once."""
    # Ensure model_name is correct and can be downloaded
    return TextSummarizer(model_name="facebook/bart-large-cnn")

summarizer_instance = load_summarizer()

st.title("💡 AI Text Summarizer")
st.markdown("Enter your text below and get a concise summary!")

if not summarizer_instance.summarizer: # Check if the model failed to load
    st.error("Failed to load the summarization model. Please check your internet connection or try again later.")
else:
    input_text = st.text_area("Paste your text here:", height=300, max_chars=5000)

    # Sliders for customization
    max_len = st.slider("Maximum summary length", 50, 500, 150)
    min_len = st.slider("Minimum summary length", 10, 200, 40)
    do_sample = st.checkbox("Enable creative sampling (may vary results)", value=False)


    if st.button("Summarize Text"):
        if input_text:
            with st.spinner("Generating summary..."):
                summary = summarizer_instance.summarize_text(
                    input_text,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=do_sample
                )
                st.subheader("Generated Summary:")
                st.success(summary)
        else:
            st.warning("Please enter some text to summarize.")

st.markdown("""
<hr>
<p style='text-align: center;'>Powered by Hugging Face Transformers and Streamlit</p>
""", unsafe_allow_html=True)