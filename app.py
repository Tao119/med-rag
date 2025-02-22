import streamlit as st
import os
import json
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from embedding import Embeddings
from tqdm import tqdm


load_dotenv()

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

DATA_DIR = "./data"
DB_DIR = "./db"
HISTORY_FILE = "./history.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


def update_db(chunk_size, embedding_model, hf_token):
    loader = DirectoryLoader(DATA_DIR, glob="*.txt")
    documents = loader.load()

    if not documents:
        st.warning("No documents found to update the database.")
        return

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = Embeddings(model_name=embedding_model, hf_token=hf_token)

    Chroma.from_documents(
        tqdm(docs),
        embeddings,
        persist_directory=DB_DIR,
    )

    st.success(
        f"✅ {len(docs)} documents were vectorized and database updated.")


def file_management_page():
    st.title("File Management")

    embedding_model = st.sidebar.text_input(
        "Embedding model name", os.getenv("HUGGINGFACE_EMBEDDING_MODEL_NAME"))
    hf_token = st.sidebar.text_input("HuggingFace API token", os.getenv(
        "HUGGINGFACE_API_TOKEN"), type="password")
    chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1000)

    st.header("Upload New File")
    uploaded_file = st.file_uploader("Upload .txt files", type=["txt"])

    if uploaded_file is not None:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"{uploaded_file.name} has been uploaded.")
        update_db(chunk_size, embedding_model, hf_token)

    st.header("Current Files")
    files = os.listdir(DATA_DIR)
    if files:
        for file in files:
            file_path = os.path.join(DATA_DIR, file)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button(f"Delete", key=file):
                    os.remove(file_path)
                    st.warning(f"{file} has been deleted.")
                    update_db(chunk_size, embedding_model, hf_token)
                    st.rerun()
    else:
        st.write("No files available.")


def question_page():
    st.title("RAG Question Page")

    k = st.sidebar.slider("Documents to retrieve (k)", 1, 10, 4)
    score_threshold = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.75)
    embedding_model = st.sidebar.text_input(
        "Embedding model name", os.getenv("HUGGINGFACE_EMBEDDING_MODEL_NAME"))
    hf_token = st.sidebar.text_input("HuggingFace API token", os.getenv(
        "HUGGINGFACE_API_TOKEN"), type="password")

    st.header("System Prompt")
    system_prompt = st.text_area("Define system instructions",
                                 value="あなたは優秀な医療アシスタントです。ユーザーの質問に対して、正確で信頼性の高い医療知識を基に分かりやすく回答してください。専門用語を使用する場合は、一般の人でも理解できるように説明してください。")

    query = st.text_input("Enter your query", "")

    if st.button("Submit") and query:
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=Embeddings(
            model_name=embedding_model, hf_token=hf_token))

        retriever = vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                "k": k
            }
        )

        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            azure_deployment=azure_deployment_name,
            api_key=azure_api_key,
            model="gpt-4o"
        )

        # Create Prompt Template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "質問: {question}\n\n関連する情報: {context}\n\n回答:")
        ])

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": prompt_template,
                "document_variable_name": "context"
            }
        )

        response = qa_chain.invoke({"query": query})
        answer = response.get("result", "No valid response found.")

        # Display Response
        st.subheader("Response")
        st.markdown(
            f"<div style='border:1px solid #d3d3d3; padding: 10px; border-radius: 5px;'>{answer}</div>",
            unsafe_allow_html=True
        )

        # Save to History
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "system_prompt": system_prompt,
            "settings": {
                "k": k,
                "score_threshold": score_threshold,
                "embedding_model": embedding_model
            },
            "response": answer
        }
        save_history(history_entry)

        # Retrieved Docs
        with st.expander("Retrieved Documents with Scores", expanded=False):
            retrieved_docs = retriever.invoke(query)
            for i, doc in enumerate(retrieved_docs):
                st.write(f"**Document {i + 1}:**")
                st.write(doc.page_content)
                st.write(
                    f"Relevance score: {doc.metadata.get('relevance_score', 'N/A')}")


def history_page():
    st.title("Query History")

    history = load_history()
    if not history:
        st.write("No query history found.")
        return

    for idx, entry in enumerate(reversed(history)):
        with st.expander(f"Query at {entry['timestamp']}"):
            st.write(f"**Query:** {entry['query']}")
            st.write(f"**System Prompt:** {entry['system_prompt']}")
            st.write(f"**Settings:**")
            st.write(f"- Documents to retrieve (k): {entry['settings']['k']}")
            st.write(
                f"- Score threshold: {entry['settings']['score_threshold']}")
            st.write(
                f"- Embedding model: {entry['settings']['embedding_model']}")
            st.write("**Response:**")
            st.markdown(
                f"<div style='border:1px solid #d3d3d3; padding: 10px; border-radius: 5px;'>{entry['response']}</div>",
                unsafe_allow_html=True
            )

            if st.button("Delete", key=f"delete_{idx}"):
                history.pop(len(history) - idx - 1)
                with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=4)
                st.success("Entry deleted.")
                st.rerun()


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["RAG Question Page", "File Management", "Query History"])

    if page == "File Management":
        file_management_page()
    elif page == "RAG Question Page":
        question_page()
    elif page == "Query History":
        history_page()


if __name__ == "__main__":
    main()
