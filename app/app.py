import streamlit as st
import json

# Load data
with open("kalenjin_data.json", "r", encoding="utf-8") as f:
    kalenjin_data = json.load(f)

st.set_page_config(page_title="Learn Kalenjin", layout="centered")

st.title("📘 Learn Kalenjin Language")
st.markdown("Discover Kalenjin words, meanings, and usage examples.")

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
        st.markdown(f"""
        ### 🗣️ {word['word'].capitalize()}
        - **Type:** *{word['type']}*
        - **Translation:** {word['translation']}
        - **Example:** _{word['example']}_
        """)
else:
    st.warning("No matching words found.")

st.markdown("---")
st.caption("Kalenjin Learning Tool | Built with ❤️ using Streamlit")
