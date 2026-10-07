"""Chat page for the assistant. Run from the repo root: streamlit run assistant/app.py"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `import assistant` work

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from assistant.agent import AgentResult, build_agent  # noqa: E402

st.set_page_config(page_title="PL Analyst Assistant", layout="centered")

EXAMPLES = [
    "How many points did Arsenal win in 2023/24?",
    "Which teams won more than 80 points in 2018/19?",
    "What are the chances that Liverpool beat Chelsea at home?",
    "Who is most likely to be relegated this season?",
    "What happened in Chelsea's 2022/23 season?",
]


@st.cache_resource(show_spinner="Loading the model, the database and the search index")
def get_agent():
    return build_agent()


def result_table(r: AgentResult) -> pd.DataFrame:
    """Query rows as a table. Column names are made unique because the page requires that."""
    names = [c if r.columns.count(c) == 1 else f"{c}_{i + 1}" for i, c in enumerate(r.columns)]
    rows = [[float(v) if isinstance(v, Decimal) else v for v in row] for row in r.rows]
    return pd.DataFrame(rows, columns=names)


def show_details(r: AgentResult) -> None:
    with st.expander("How this was answered"):
        st.write(
            f"Tool: `{r.tool or 'none'}`. Time: {r.seconds:.0f} s. Model calls used "
            f"{r.prompt_tokens} prompt tokens and {r.output_tokens} output tokens."
        )
        if r.sql:
            st.code(r.sql, language="sql")
        if r.rows:
            st.dataframe(result_table(r), hide_index=True)
        for h in r.sources:
            st.markdown(f"**[{h.rank}]** [{h.title}]({h.url}), section {h.section}")
            st.caption(h.text[:400])
        if r.sources:
            st.caption("Article text from Wikipedia, licensed CC BY-SA 4.0.")
            if not r.citations_ok:
                st.warning("The answer does not cite its sources correctly. Check it yourself.")


with st.sidebar:
    st.header("About")
    st.write(
        "This assistant answers Premier League questions from three places: the project's "
        "results warehouse, its Dixon-Coles match model, and Wikipedia season articles."
    )
    st.write(
        "A small local language model chooses the tool and words the answer. It makes "
        "mistakes. Open **How this was answered** under each reply to check the query, "
        "the rows and the sources."
    )
    st.write("Each question is answered on its own. Write every question in full.")
    st.subheader("Try one")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state.pending = example

st.title("Premier League Analyst Assistant")
st.caption(
    "Results and xG: football-data.co.uk and Understat. Article text: Wikipedia, CC BY-SA 4.0."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("result"):
            show_details(message["result"])

typed = st.chat_input("Ask about results, predictions or a club's season")
question = typed or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Working. A local model can take a minute."):
            result = get_agent().ask(question)
        st.markdown(result.answer)
        show_details(result)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "result": result}
    )