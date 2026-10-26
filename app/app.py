import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from translate import DIRECTIONS, Translator

# Load data
with open(os.path.join(os.path.dirname(__file__), "kalenjin_data.json"), "r", encoding="utf-8") as f:
    kalenjin_data = json.load(f)

st.set_page_config(page_title="Learn Kalenjin", layout="centered")

st.title("📘 Learn Kalenjin Language")


@st.cache_resource
def get_translator(direction):
    return Translator(direction)


translate_tab, dictionary_tab = st.tabs(["Translate", "Dictionary"])

with translate_tab:
    direction = st.radio(
        "Direction", ["kln-en", "en-kln"],
        format_func=lambda d: "Kalenjin → English" if d == "kln-en" else "English → Kalenjin",
        horizontal=True,
    )
    text = st.text_area("Text to translate")
    if st.button("Translate") and text.strip():
        if not os.path.isdir(DIRECTIONS[direction][0]):
            st.error(f"Model '{DIRECTIONS[direction][0]}' not found. Train it first (see README).")
        else:
            with st.spinner("Translating..."):
                st.success(get_translator(direction)(text))

dictionary_tab.markdown("Discover Kalenjin words, meanings, and usage examples.")
with dictionary_tab:
    # Search & filter
    search_query = st.text_input("🔍 Search Kalenjin word:")
    word_type = st.selectbox("Filter by word type", ["All"] + sorted(set(w["type"] for w in kalenjin_data)))

    # Filter logic
    filtered_words = [
        w for w in kalenjin_data
        if (search_query.lower() in w["word"].lower())
        and (word_type == "All" or w["type"] == word_type)
    ]

    # Display
    if filtered_words:
        for word in filtered_words:
            st.markdown(
                f"### 🗣️ {word['word'].capitalize()}\n\n"
                f"- **Type:** *{word['type']}*\n"
                f"- **Translation:** {word['translation']}\n"
                f"- **Example:** _{word['example']}_"
            )
    else:
        st.warning("No matching words found.")

st.markdown("---")
st.caption("Kalenjin Learning Tool | Built with ❤️ using Streamlit")
