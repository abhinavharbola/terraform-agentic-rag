import streamlit as st

from src.clients import qdrant_client
from src.config import settings
from src.graph import run_turn

st.set_page_config(page_title="Terraform RAG", page_icon="◧", layout="centered")

CUSTOM_CSS = """
<style>
.stApp [data-testid="stChatMessage"] { gap: 0.6rem; }

code, .mono { font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace; }

.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.35rem 0 0.15rem 0; }
.chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
    font-size: 0.72rem; padding: 0.15rem 0.55rem; border-radius: 999px;
    border: 1px solid rgba(61, 174, 160, 0.35);
    background: rgba(61, 174, 160, 0.08); color: #2E8B7F;
}
.chip.miss { border-color: rgba(150, 150, 150, 0.35); background: rgba(150, 150, 150, 0.08); color: #888; }
.chip.refusal { border-color: rgba(217, 119, 87, 0.4); background: rgba(217, 119, 87, 0.08); color: #B5573D; }
.chip.missing { border-color: rgba(192, 57, 43, 0.4); background: rgba(192, 57, 43, 0.08); color: #C0392B; }

.source-row {
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
    font-size: 0.8rem; padding: 0.3rem 0; border-bottom: 1px solid rgba(128,128,128,0.15);
    display: flex; justify-content: space-between; gap: 1rem;
}
.source-row .score { color: #2E8B7F; white-space: nowrap; }
.source-row .badge-official { color: #2E8B7F; font-size: 0.7rem; }
.source-row .badge-community { color: #888; font-size: 0.7rem; }

.example-btn button { text-align: left !important; }

/* --- sidebar --- */
[data-testid="stSidebar"] { border-right: 1px solid rgba(61, 174, 160, 0.15); }
.eyebrow {
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
    font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #6b7280; margin: 0.2rem 0 0.5rem 0;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
    font-size: 1.4rem; color: #2E8B7F;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.72rem; color: #9ca3af; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

EXAMPLE_QUESTIONS = [
    "How do I pin a provider version?",
    "What's the difference between count and for_each?",
    "How do I safely destroy just one resource?",
    "How do I set up a remote state backend?",
]


@st.cache_data(ttl=30)
def get_corpus_stats():
    try:
        docs = qdrant_client.count(collection_name=settings.qdrant_docs_collection, exact=False).count
        cached = qdrant_client.count(collection_name=settings.qdrant_cache_collection, exact=False).count
        return docs, cached
    except Exception:
        return None, None


def get_session_stats():
    assistant_turns = [t for t in st.session_state.history if t["role"] == "assistant"]
    total = len(assistant_turns)
    if total == 0:
        return 0, "—"
    cache_hits = sum(1 for t in assistant_turns if t.get("details", {}).get("cache_layer"))
    return total, f"{round(cache_hits / total * 100)}%"


with st.sidebar:
    st.markdown('<div class="eyebrow">Corpus</div>', unsafe_allow_html=True)
    doc_count, cache_count = get_corpus_stats()
    corpus_col1, corpus_col2 = st.columns(2)
    corpus_col1.metric("Chunks indexed", doc_count if doc_count is not None else "—")
    corpus_col2.metric("Cached answers", cache_count if cache_count is not None else "—")

    st.markdown('<div class="eyebrow" style="margin-top: 1rem;">This session</div>', unsafe_allow_html=True)
    questions_asked, hit_rate = get_session_stats()
    session_col1, session_col2 = st.columns(2)
    session_col1.metric("Questions asked", questions_asked)
    session_col2.metric("Cache hit rate", hit_rate)

    st.markdown('<div class="eyebrow" style="margin-top: 1rem;">Connections</div>', unsafe_allow_html=True)
    key_fields = {
        "NIM": settings.nvidia_nim_api_key,
        "Groq": settings.groq_api_key,
        "Gemini": settings.gemini_api_key,
        "Qdrant": settings.qdrant_url and settings.qdrant_api_key,
    }
    chips = [
        f'<span class="chip{"" if value else " missing"}">{label}</span>' for label, value in key_fields.items()
    ]
    st.markdown(f'<div class="chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        get_corpus_stats.clear()
        st.rerun()

st.markdown("### Terraform Q&A")
st.caption("Grounded in your ingested official and community docs. Off-topic questions are declined.")


def render_details(details: dict) -> None:
    cache_layer = details.get("cache_layer")
    provider = details.get("provider")
    is_refusal = details.get("refusal", False)

    chips = []
    if is_refusal:
        chips.append('<span class="chip refusal">declined</span>')
    elif cache_layer:
        chips.append(f'<span class="chip">cache: {cache_layer}</span>')
    else:
        chips.append('<span class="chip miss">cache: miss</span>')
        if provider:
            chips.append(f'<span class="chip">served by: {provider}</span>')
        sources = details.get("sources") or []
        if sources:
            chips.append(f'<span class="chip">{len(sources)} source{"s" if len(sources) != 1 else ""}</span>')

    st.markdown(f'<div class="chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)

    sources = details.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                path = source["metadata"].get("source_path", "unknown")
                authority = source["metadata"].get("source_authority", "unknown")
                badge_class = "badge-official" if authority == "official" else "badge-community"
                st.markdown(
                    f'<div class="source-row"><span>{path} '
                    f'<span class="{badge_class}">[{authority}]</span></span>'
                    f'<span class="score">{source["rerank_score"]:.3f}</span></div>',
                    unsafe_allow_html=True,
                )


if not st.session_state.history:
    st.markdown("&nbsp;")
    st.caption("Try one of these, or ask your own below.")
    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
            st.markdown("</div>", unsafe_allow_html=True)

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("details"):
            render_details(turn["details"])

prompt = st.chat_input("Ask a Terraform question")
if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            plain_history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.history[:-1]]
            result = run_turn(prompt, plain_history)
        st.markdown(result["answer"])

        details = {
            "cache_layer": result.get("cache_layer"),
            "provider": result.get("provider"),
            "sources": result.get("reranked") if not result.get("cache_layer") else None,
            "refusal": not result.get("allowed", True),
        }
        render_details(details)

    st.session_state.history.append({"role": "assistant", "content": result["answer"], "details": details})
    get_corpus_stats.clear()