"""Streamlit frontend for the RAG system."""

from __future__ import annotations

import os
from typing import Any, Dict

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")


def _upload_document(uploaded_file) -> Dict[str, Any]:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def _query_documents(question: str, filters: Dict[str, Any], conversation_id: str | None) -> Dict[str, Any]:
    payload = {
        "question": question,
        "top_k": 5,
        "use_hybrid": True,
        "use_reranking": True,
        "filters": filters,
        "conversation_id": conversation_id,
        "stream": False,
    }
    response = requests.post(f"{API_URL}/query", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def _list_documents() -> list[dict[str, Any]]:
    response = requests.get(f"{API_URL}/documents", timeout=60)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="RAG Assistant", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #0f172a 0%, #111827 100%); color: #e5e7eb; }
      .block-container { padding-top: 1.5rem; }
      .panel { background: rgba(17, 24, 39, 0.9); border: 1px solid rgba(148, 163, 184, 0.15); padding: 1rem; border-radius: 16px; }
      .source-card { background: rgba(30, 41, 59, 0.9); padding: 0.8rem; border-radius: 12px; margin-bottom: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

st.title("Retrieval-Augmented Assistant")
left, right = st.columns([2.2, 1])

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF, TXT, Markdown, or DOCX",
        type=["pdf", "txt", "md", "markdown", "docx"],
    )
    if uploaded and st.button("Index Document", use_container_width=True):
        with st.spinner("Extracting, chunking, and indexing..."):
            result = _upload_document(uploaded)
        st.success(f"Indexed {result['document']['filename']} with {result['document']['chunk_count']} chunks.")

    st.divider()
    st.subheader("Indexed Files")
    try:
        for item in _list_documents():
            st.caption(f"{item['filename']} ({item['chunk_count']} chunks)")
    except Exception as exc:
        st.warning(f"API unavailable: {exc}")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a grounded question about your documents")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating answer..."):
                response = _query_documents(prompt, {}, st.session_state.conversation_id)
                st.session_state.conversation_id = response["conversation_id"]
                st.markdown(response["answer"])
                st.caption(f"Confidence score: {response['confidence']:.3f}")
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        st.session_state["latest_response"] = response
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Retrieved Sources")
    latest = st.session_state.get("latest_response")
    if latest:
        rewritten_query = latest.get("rewritten_query")
        if rewritten_query:
            st.caption(f"Rewritten query: {rewritten_query}")
        for citation in latest.get("citations", []):
            st.markdown('<div class="source-card">', unsafe_allow_html=True)
            st.markdown(f"**{citation['filename']}**")
            page = citation.get("page_number")
            if page:
                st.caption(f"Page {page}")
            score = citation.get("score")
            if score is not None:
                st.caption(f"Score: {score:.3f}")
            st.write(citation["snippet"])
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Retrieved chunks and citations will appear here after your first question.")
    st.markdown("</div>", unsafe_allow_html=True)
