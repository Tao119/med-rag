import streamlit as st
from utils.settings_utils import load_settings, save_settings
from utils.db_utils import save_history
from datetime import datetime
import os
from embedding import Embeddings
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.retrievers.ensemble import EnsembleRetriever


load_dotenv()

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")


def load_all_vector_dbs(db_base_dir, embedding_model, hf_token):
    vector_dbs = []

    for db_name in os.listdir(db_base_dir):
        db_path = os.path.join(db_base_dir, db_name)
        if os.path.isdir(db_path):
            vector_db = Chroma(
                persist_directory=db_path,
                embedding_function=Embeddings(
                    model_name=embedding_model, hf_token=hf_token
                ),
            )
            vector_dbs.append(vector_db)

    return vector_dbs


from langchain.schema import BaseRetriever, Document
from typing import List
from pydantic import BaseModel


class LimitedEnsembleRetriever(BaseRetriever, BaseModel):
    retrievers: List[BaseRetriever]
    k: int

    def get_relevant_documents(self, query: str) -> List[Document]:
        all_results = []
        for retriever in self.retrievers:
            docs = retriever.get_relevant_documents(query)
            all_results.extend(docs)

        sorted_results = sorted(
            all_results,
            key=lambda doc: doc.metadata.get("relevance_score", 0),
            reverse=True,
        )

        return sorted_results[: self.k]


def merge_retrievers(vector_dbs, score_threshold, k):
    retrievers = [
        db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": k},
        )
        for db in vector_dbs
    ]

    ensemble_retriever = LimitedEnsembleRetriever(retrievers=retrievers, k=k)

    return ensemble_retriever


def possible_models(source: str):
    all_models = {
        "OpenAI": [
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-instruct-0914",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-instruct",
        ],
        "Azure": ["gpt-4o"],
        "vllm": ["rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b"],
    }

    return all_models.get(source, [])


def question_page(user_path, username):
    settings = load_settings(user_path)
    st.title("RAG Question Page")

    # --- Sidebar Settings ---
    st.sidebar.header("Settings")

    source_list = [
        "OpenAI",
        "Azure",
        "vllm",
    ]
    default_source = settings.get("source", "OpenAI")

    model_source =st.sidebar.selectbox(
        "Model Source",
        options=source_list,
        index=source_list.index(default_source) if default_source in source_list else 0,
        key="source_q",
    )

    model_list = possible_models(model_source)
    default_model = settings.get("chat_model", model_list[0])

    chat_model = st.sidebar.selectbox(
        "Chat model name",
        options=model_list,
        index=model_list.index(default_model) if default_model in model_list else 0,
        key="chat_model_q",
    )

    k = st.sidebar.slider(
        "Documents to retrieve (k)", 1, 20, settings.get("k"), key="k_slider"
    )

    st.sidebar.write("Score threshold")
    score_threshold = st.sidebar.slider(
        " ", 0.0, 1.0, settings.get("score_threshold"), step=0.01, key="score_slider"
    )

    # Embedding model
    embedding_model = st.sidebar.text_input(
        "Embedding model name", settings.get("embedding_model"), key="embedding_model_q"
    )

    # HuggingFace token
    hf_token = st.sidebar.text_input(
        "HuggingFace API token",
        settings.get("hf_token"),
        type="password",
        key="hf_token_q",
    )

    # Save Settings Button
    if st.sidebar.button("Save Settings"):
        settings.update(
            {
                "chat_model": chat_model,
                "k": k,
                "score_threshold": score_threshold,
                "embedding_model": embedding_model,
                "hf_token": hf_token,
            }
        )
        save_settings(user_path, settings)
        st.success("Settings saved successfully!")

    # --- Main Content ---
    st.header("System Prompt")
    system_prompt = st.text_area(
        "Define system instructions",
        value=settings["system_prompt"],
        key="system_prompt",
    )

    query = st.text_input("Enter your query", "")

    if st.button("Submit") and query:
        # Vector DB Initialization
        vector_dbs = load_all_vector_dbs(
            os.path.join(user_path, "db"), embedding_model, hf_token
        )

        retriever = merge_retrievers(vector_dbs, score_threshold, k)

        if model_source == "OpenAI":
            llm = ChatOpenAI(api_key=openai_api_key, model=chat_model)
        elif model_source == "Azure":
            llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                azure_deployment=settings.get(
                    "chat_model", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                ),
                api_key=azure_api_key,
                model=chat_model,
            )
        else:
            llm = ChatOpenAI(base_url="http://localhost:8000", model=chat_model)


        # Prompt Template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "質問: {question}\n\n関連する情報: {context}\n\n回答:"),
            ]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": prompt_template,
                "document_variable_name": "context",
            },
        )

        # Query Execution
        response = qa_chain.invoke({"query": query})
        answer = response.get("result", "No valid response found.")

        # Display Response
        st.subheader("Response")
        st.markdown(
            f"<div style='border:1px solid #d3d3d3; padding: 10px; border-radius: 5px;'>{answer}</div>",
            unsafe_allow_html=True,
        )

        # Display Retrieved Documents
        with st.expander("Retrieved Documents with Scores", expanded=False):
            retrieved_docs = retriever.invoke(query)
            for i, doc in enumerate(retrieved_docs):
                document_name = doc.metadata.get("source", "Unknown Document")
                chunk_number = doc.metadata.get("chunk_number", "N/A")
                st.write(f"**Document {i + 1}:**")
                st.write(f"**Source:** {document_name} | **Chunk:** {chunk_number}")
                st.write(
                    f"Relevance score: {doc.metadata.get('relevance_score', 'N/A')}"
                )
                st.write(doc.page_content)

        # Save Query History
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "system_prompt": system_prompt,
            "settings": {
                "k": k,
                "score_threshold": score_threshold,
                "embedding_model": embedding_model,
                "chat_model": chat_model,
            },
            "response": answer,
            "username": username,
            "retrieved_docs": [
                {
                    "content": str(doc.page_content),
                    "score": doc.metadata.get("relevance_score", "N/A"),
                    "document_name": doc.metadata.get("source", "Unknown Document"),
                    "chunk_number": doc.metadata.get("chunk_number", "N/A"),
                }
                for doc in retrieved_docs
            ],
        }
        save_history(user_path, history_entry)

        st.success("Query executed and history saved successfully.")
