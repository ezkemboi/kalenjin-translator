import os
import base64
import difflib
import io
import random
import sys
import threading
import uuid
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from translate import DIRECTIONS, Translator

st.set_page_config(page_title="Learn Kalenjin", layout="centered")

st.title("Kalenjin Language & Dialects")


@st.cache_resource
def get_translator(direction):
    return Translator(direction)


FEEDBACK_PATH = "model/data/feedback.csv"
VOTES_PATH = "model/data/word_votes.csv"
QA_PATH = "model/data/qa_answers.csv"
QUIZ_LEN = 5
DIALECTS = ["Marakwet", "Nandi", "Kipsigis", "Keiyo", "Tugen", "Pokot", "Other"]


def client_info():
    """IP and user agent for future spam filtering (IP is the first X-Forwarded-For hop behind a proxy)."""
    try:
        h = st.context.headers
        ip = (h.get("X-Forwarded-For") or "").split(",")[0].strip() or getattr(st.context, "ip_address", None) or ""
        return {"ip": ip, "user_agent": h.get("User-Agent", "")}
    except Exception:
        return {"ip": "", "user_agent": ""}


def _secret(name):
    try:
        return os.environ.get(name) or st.secrets.get(name)
    except Exception:
        return os.environ.get(name)


def _push_to_github(kind, row):
    """Store each submission as its own file in the private dataset repo (no write conflicts)."""
    token, repo = _secret("GITHUB_TOKEN"), _secret("DATASET_REPO")
    if not (token and repo):
        return
    buf = io.StringIO()
    pd.DataFrame([row]).to_csv(buf, index=False)
    name = f"data/{kind}/{datetime.now():%Y-%m-%d}/{datetime.now():%H%M%S}-{uuid.uuid4().hex[:8]}.csv"
    try:
        requests.put(
            f"https://api.github.com/repos/{repo}/contents/{name}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={"message": f"add {kind}", "content": base64.b64encode(buf.getvalue().encode()).decode()},
            timeout=20,
        )
    except requests.RequestException:
        pass  # the local CSV copy below is the fallback


def save_row(path, row):
    row = {"time": datetime.now().isoformat(timespec="seconds"), **row, **client_info()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    kind = os.path.splitext(os.path.basename(path))[0]
    threading.Thread(target=_push_to_github, args=(kind, row), daemon=True).start()


translate_tab, dictionary_tab, quiz_tab = st.tabs(["Translate", "Dictionary", "Quiz"])

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
            st.session_state.pop("last", None)
        else:
            with st.spinner("Translating..."):
                result = get_translator(direction)(text)
            st.session_state["last"] = {"direction": direction, "source": text.strip(), "output": result}

    last = st.session_state.get("last")
    if last:
        st.success(last["output"])
        with st.form("feedback", clear_on_submit=True):
            st.markdown("**Missing or wrong? Tell us the right meaning.**")
            correction = st.text_input("Correct translation / meaning")
            c1, c2 = st.columns(2)
            dialect = c1.selectbox("Your dialect", DIALECTS)
            sub_dialect = c2.text_input("Sub-dialect / area (optional)")
            note = st.text_input("Note (optional, e.g. spelling, context)")
            if st.form_submit_button("Save feedback") and correction.strip():
                save_row(FEEDBACK_PATH, {
                    "direction": last["direction"],
                    "source": last["source"],
                    "model_output": last["output"],
                    "correction": correction.strip(),
                    "dialect": dialect,
                    "sub_dialect": sub_dialect.strip(),
                    "note": note.strip(),
                })
                st.success("Thank you, saved.")

DICT_PATH = "model/data/dictionary.csv"


@st.cache_data
def load_dictionary():
    return pd.read_csv(DICT_PATH).fillna("") if os.path.exists(DICT_PATH) else None


with dictionary_tab:
    learned = load_dictionary()
    if learned is None:
        st.info("Build the learned dictionary first: python model/build_dictionary.py")
    else:
        st.markdown("Words learned automatically from the aligned Bible verses. Choose the language you are searching from.")
        lang = st.radio("Search in:", ["Kalenjin", "English"], horizontal=True)
        col = "kalenjin" if lang == "Kalenjin" else "english"
        query = st.text_input(f"🔍 Search a {lang} word:").strip().lower()
        if query:
            hits = learned[learned[col] == query].sort_values("score", ascending=False)
            if hits.empty:
                hits = learned[learned[col].str.startswith(query)].sort_values("score", ascending=False).head(30)
                if not hits.empty:
                    st.caption(f"No exact match for '{query}'. Showing {lang} words that start with it.")
            if hits.empty:
                st.warning(f"'{query}' was not learned from the Bible text (rare or absent word).")
            for _, r in hits.iterrows():
                st.markdown(f"### 🗣️ {r.kalenjin} → {r.english}")
                st.caption(f"confidence {r.score} · seen in {r.pair_count} verses")
                if r.example_kln:
                    st.markdown(f"_{r.example_kln}_  \n{r.example_en}")
                with st.expander("Is this right in your dialect?"):
                    with st.form(f"vote_{r.name}", clear_on_submit=True):
                        vote = st.radio("Your answer", ["Yes, correct", "No, different meaning", "Not sure"], horizontal=True)
                        c1, c2 = st.columns(2)
                        dialect = c1.selectbox("Your dialect", DIALECTS, key=f"d{r.name}")
                        sub_dialect = c2.text_input("Sub-dialect / area (optional)", key=f"s{r.name}")
                        suggestion = st.text_input("If different: what does it mean / what do you say?", key=f"g{r.name}")
                        if st.form_submit_button("Submit"):
                            if vote.startswith("No") and not suggestion.strip():
                                st.warning("Please add what it means in your dialect.")
                            else:
                                save_row(VOTES_PATH, {
                                    "kalenjin": r.kalenjin, "english": r.english,
                                    "vote": {"Y": "agree", "N": "disagree"}.get(vote[0], "unsure"),
                                    "dialect": dialect, "sub_dialect": sub_dialect.strip(),
                                    "suggestion": suggestion.strip(),
                                })
                                st.success("Thank you, saved.")

def build_quiz(df):
    """Mix of known-answer multiple choice (confident entries) and open questions (uncertain entries)."""
    known_pool = df[(df.score >= 0.4) & (df.pair_count >= 10)]
    # skip proper names, which look nearly the same in both languages
    known_pool = known_pool[[difflib.SequenceMatcher(None, k, e).ratio() < 0.6 for k, e in zip(known_pool.kalenjin, known_pool.english)]]
    open_pool = df[(df.score < 0.25) & (df.pair_count >= 3)]
    n_known = min(round(QUIZ_LEN * 0.7), len(known_pool))
    n_open = min(QUIZ_LEN - n_known, len(open_pool))
    questions = []
    for _, r in known_pool.sample(n_known).iterrows():
        # a word can have several valid meanings (e.g. iman: surely, righteous, verily)
        accepted = df[(df.kalenjin == r.kalenjin) & (df.score >= 0.1)].english.tolist()
        wrong = df[~df.english.isin(df[df.kalenjin == r.kalenjin].english)].english.drop_duplicates().sample(3).tolist()
        options = wrong + [r.english]
        random.shuffle(options)
        questions.append({"type": "known", "kalenjin": r.kalenjin, "english": r.english,
                          "accepted": accepted, "options": options})
    for _, r in open_pool.sample(n_open).iterrows():
        questions.append({"type": "open", "kalenjin": r.kalenjin, "english": r.english, "options": []})
    random.shuffle(questions)
    return questions


with quiz_tab:
    qdf = load_dictionary()
    if qdf is None or len(qdf) < 50:
        st.info("Build the learned dictionary first: python model/build_dictionary.py")
    else:
        st.markdown("Test your Kalenjin and help us improve it. Some questions have a known answer, others we are still unsure about, so your answer matters either way.")
        c1, c2 = st.columns(2)
        q_dialect = c1.selectbox("Your dialect", DIALECTS, key="q_dialect")
        q_sub = c2.text_input("Sub-dialect / area (optional)", key="q_sub")

        if "quiz" not in st.session_state:
            st.session_state.update(quiz=build_quiz(qdf), qi=0, score=0, scored=0)
        quiz = st.session_state["quiz"]
        qi = st.session_state["qi"]

        if qi >= len(quiz):
            st.success(f"Round finished. You got {st.session_state['score']} of {st.session_state['scored']} scored questions right. Thank you for your answers!")
            if st.button("New round"):
                for k in ("quiz", "qi", "score", "scored"):
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            q = quiz[qi]
            st.progress(qi / len(quiz), text=f"Question {qi + 1} of {len(quiz)}")
            with st.form(f"quiz_{qi}"):
                if q["type"] == "known":
                    st.markdown(f"### What does **{q['kalenjin']}** mean?")
                    answer = st.radio("Choose one", q["options"], index=None)
                else:
                    st.markdown(f"### What does **{q['kalenjin']}** mean in your dialect?")
                    st.caption("We are not sure about this one. Type it in English, or write 'don't know'.")
                    answer = st.text_input("Your answer")
                submitted = st.form_submit_button("Submit")
            if submitted:
                if not answer or not answer.strip():
                    st.warning("Please answer first.")
                else:
                    correct = ""
                    if q["type"] == "known":
                        correct = answer in q["accepted"]
                        st.session_state["scored"] += 1
                        st.session_state["score"] += int(correct)
                    save_row(QA_PATH, {
                        "question_type": q["type"], "kalenjin": q["kalenjin"],
                        "expected": q["english"] if q["type"] == "known" else "",
                        "answer": answer.strip(), "correct": correct,
                        "dialect": q_dialect, "sub_dialect": q_sub.strip(),
                    })
                    st.session_state["qi"] += 1
                    st.session_state["flash"] = (
                        (f"Correct! Other meanings: {', '.join(m for m in q['accepted'] if m != answer)}." if len(q["accepted"]) > 1 else "Correct!") if correct is True else
                        f"Not quite. It is mostly **{q['english']}**." if correct is False else
                        "Thank you, saved."
                    )
                    st.rerun()
            if st.session_state.get("flash"):
                st.info(st.session_state.pop("flash"))

st.markdown("---")
st.caption("Kalenjin Learning Tool | Built by Ezrqn Kemboi with ❤️ using Streamlit")
