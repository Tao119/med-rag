import streamlit as st
import json
import os
from utils.db_utils import load_history, load_global_history
from datetime import datetime


def history_page(user_path):
    st.title("Query History")

    tabs = st.tabs(["My History", "Global History"])

    with tabs[0]:
        st.subheader("My Query History")
        user_history = load_history(user_path)

        if not user_history:
            st.write("No query history found.")
        else:
            user_history_sorted = sorted(
                user_history, key=lambda x: datetime.fromisoformat(x['timestamp']))

            for idx, entry in enumerate(user_history_sorted):
                with st.expander(f"Query at {entry['timestamp']}"):
                    st.write(f"**Query:** {entry['query']}")
                    st.write(f"**System Prompt:** {entry['system_prompt']}")
                    st.write("**Settings:**")
                    st.write(
                        f"- Documents to retrieve (k): {entry['settings']['k']}")
                    st.write(
                        f"- Score threshold: {entry['settings']['score_threshold']}")
                    st.write(
                        f"- Embedding model: {entry['settings']['embedding_model']}")
                    st.write("**Response:**")
                    st.markdown(
                        f"<div style='border:1px solid #d3d3d3; padding: 10px; border-radius: 5px;'>{entry['response']}</div>",
                        unsafe_allow_html=True
                    )
                    retrieved_docs = entry.get('retrieved_docs', [])
                    if retrieved_docs:
                        st.write("### Retrieved Documents with Scores")
                        for i, doc in enumerate(retrieved_docs):
                            st.write(f"**Document {i + 1}:**")
                            st.write(doc.get('content', 'No content'))
                            st.write(
                                f"Relevance score: {doc.get('score')}")

                    if st.button("Delete", key=f"user_delete_{idx}"):
                        user_history.remove(entry)
                        history_file = os.path.join(user_path, "history.json")
                        with open(history_file, "w", encoding="utf-8") as f:
                            json.dump(user_history, f,
                                      ensure_ascii=False, indent=4)
                        st.success("Entry deleted.")
                        st.rerun()

    with tabs[1]:
        st.subheader("Global Query History")
        global_history = load_global_history()

        if not global_history:
            st.write("No global query history found.")
        else:
            global_history_sorted = sorted(
                global_history, key=lambda x: datetime.fromisoformat(x['timestamp']))

            for idx, entry in enumerate(global_history_sorted):
                with st.expander(f"Query at {entry['timestamp']} by {entry.get('username', 'Unknown User')}"):
                    st.write(f"**Query:** {entry['query']}")
                    st.write(f"**System Prompt:** {entry['system_prompt']}")
                    st.write("**Settings:**")
                    st.write(
                        f"- Documents to retrieve (k): {entry['settings']['k']}")
                    st.write(
                        f"- Score threshold: {entry['settings']['score_threshold']}")
                    st.write(
                        f"- Embedding model: {entry['settings']['embedding_model']}")
                    st.write("**Response:**")
                    st.markdown(
                        f"<div style='border:1px solid #d3d3d3; padding: 10px; border-radius: 5px;'>{entry['response']}</div>",
                        unsafe_allow_html=True
                    )
