import re
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Word Occurrence Visualizer", layout="wide")

st.title("🔎 Word Occurrence Visualizer")
st.write("Paste a passage below and explore how often each word appears. Use the sidebar to sort and style the chart.")

# --- Sidebar controls ---
st.sidebar.header("Options")
case_sensitive = st.sidebar.checkbox("Case sensitive", value=False, help="If off, 'Hello' and 'hello' are treated the same.")
remove_numbers = st.sidebar.checkbox("Ignore numbers", value=True, help="Treat numbers as non-words.")

show_prepositions = st.sidebar.checkbox("Show prepositions", value=True, help="Uncheck to omit common prepositions.")

sort_mode = st.sidebar.radio(
    "Sort by",
    options=["Frequency (High → Low)", "Alphabetical (A → Z)"],
    index=0,
)

# When alphabetical, allow direction toggle
alpha_ascending = st.sidebar.checkbox("Alphabetical ascending", value=True)

# Limit how many bars to show to keep chart readable
max_bars = st.sidebar.slider("Show top N words", min_value=5, max_value=200, value=30, step=5)

bar_color = st.sidebar.color_picker("Bar color", value="#1f77b4")

# --- Input text ---
default_text = (
    """
    Streamlit makes it easy to build and share data apps. Paste any text here and this app
    will count how many times each word occurs. You can sort by frequency or alphabetically,
    and you can also change the bar color. Try it out with your own passage!
    """
)

text = st.text_area("Text input", value=default_text, height=220)

# --- Prepositions list ---
prepositions = {
    "about", "above", "across", "after", "against", "along", "among", "around", "at",
    "before", "behind", "below", "beneath", "beside", "between", "beyond", "but", "by",
    "concerning", "considering", "despite", "down", "during", "except", "for", "from",
    "in", "inside", "into", "like", "near", "of", "off", "on", "onto", "out", "outside",
    "over", "past", "regarding", "since", "through", "throughout", "to", "toward", "under",
    "underneath", "until", "up", "upon", "with", "within", "without"
}

# --- Processing ---
def tokenize(text: str, case_sensitive: bool = False, ignore_numbers: bool = True):
    if not case_sensitive:
        text = text.lower()
    # Choose regex based on whether we ignore numbers
    if ignore_numbers:
        pattern = r"[a-zA-Z]+(?:['-][a-zA-Z]+)*"
    else:
        pattern = r"[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)*"
    return re.findall(pattern, text)


def count_words(text: str, case_sensitive: bool, ignore_numbers: bool, show_prepositions: bool) -> pd.DataFrame:
    tokens = tokenize(text, case_sensitive=case_sensitive, ignore_numbers=ignore_numbers)
    if not show_prepositions:
        tokens = [t for t in tokens if t not in prepositions]
    counts = Counter(tokens)
    df = pd.DataFrame(counts.items(), columns=["word", "count"]) if counts else pd.DataFrame(columns=["word", "count"])
    return df


df = count_words(text, case_sensitive=case_sensitive, ignore_numbers=remove_numbers, show_prepositions=show_prepositions)

if df.empty:
    st.info("No words found. Paste some text to get started.")
    st.stop()

# --- Sorting & limiting ---
if sort_mode.startswith("Frequency"):
    df = df.sort_values(["count", "word"], ascending=[False, True])
else:
    df = df.sort_values("word", ascending=alpha_ascending)

# Keep only top N for chart
chart_df = df.head(max_bars)

# --- Layout: table and chart side by side ---
left, right = st.columns([1, 2])

with left:
    st.subheader("Counts table")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)

with right:
    st.subheader("Bar chart")
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color=bar_color)
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("word:N", sort=None, title="Word"),
            tooltip=["word", "count"],
        )
        .properties(height=max(250, 18 * len(chart_df)), width="container")
    )
    st.altair_chart(chart, use_container_width=True)

# --- Footer ---
st.caption(
    "Tip: Use the sidebar to switch sorting between frequency and alphabetical, toggle case sensitivity, "
    "ignore numbers, show or omit prepositions, and pick any bar color you like."
)