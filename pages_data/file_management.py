import streamlit as st
import os
from utils.settings_utils import load_settings, save_settings

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

from embedding import Embeddings
from tqdm import tqdm

DEFAULT_DATA_DIR = "./data"


def update_db(data_dir, db_dir, chunk_size, chunk_overlap, embedding_model, hf_token):
    loader = DirectoryLoader(data_dir, glob="*.txt")
    documents = loader.load()

    if not documents:
        st.warning("No documents found to update the database.")
        return

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * (chunk_overlap / 100))
    )
    docs = text_splitter.split_documents(documents)

    embeddings = Embeddings(model_name=embedding_model, hf_token=hf_token)

    Chroma.from_documents(
        tqdm(docs),
        embeddings,
        persist_directory=db_dir,
    )

    st.success(
        f"✅ {len(docs)} documents were vectorized and database updated with {chunk_overlap}% overlap."
    )


def file_management_page(user_path):
    settings = load_settings(user_path)

    st.title("File Management")

    embedding_model = st.sidebar.text_input(
        "Embedding model name", settings.get("embedding_model", ""), key="embedding_model"
    )
    hf_token = st.sidebar.text_input(
        "HuggingFace API token", settings.get("hf_token", ""), type="password", key="hf_token"
    )
    chunk_size = st.sidebar.slider(
        "Chunk size", 500, 2000, settings.get("chunk_size", 1000), key="chunk_size"
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (%)", 0, 50, settings.get("chunk_overlap", 25), key="chunk_overlap"
    )

    user_db_dir = os.path.join(user_path, "db")
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    os.makedirs(user_db_dir, exist_ok=True)

    st.sidebar.header("Data Directory Settings")
    data_dir = st.sidebar.text_input(
        "Data Directory", DEFAULT_DATA_DIR, disabled=True)
    db_dir = st.sidebar.text_input("Database Directory", user_db_dir)

    if st.sidebar.button("Save Paths"):
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        settings["data_dir"] = data_dir
        settings["db_dir"] = db_dir
        save_settings(user_path, settings)
        update_db(data_dir, db_dir, chunk_size,
                  chunk_overlap, embedding_model, hf_token)
        st.success("Paths updated successfully.")
        st.rerun()

    tab1, tab2 = st.tabs(["Upload", "Create"])

    with tab1:
        st.header("Upload New File")
        uploaded_file = st.file_uploader("Upload .txt files", type=["txt"])

        if uploaded_file is not None:
            save_path = os.path.join(DEFAULT_DATA_DIR, uploaded_file.name)

            if os.path.exists(save_path):
                st.error(
                    f"A file named '{uploaded_file.name}' already exists. Please use a different name.")
            else:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"{uploaded_file.name} has been uploaded.")
                update_db(DEFAULT_DATA_DIR, user_db_dir, chunk_size,
                          chunk_overlap, embedding_model, hf_token)
                st.rerun()

    with tab2:
        st.header("Create New File")
        new_file_name = st.text_input("Enter file name (without extension):")
        new_file_content = st.text_area("Enter file content:")

        if st.button("Create File"):
            if not new_file_name:
                st.error("Please enter a file name.")
            elif not new_file_content:
                st.error("Please enter some content for the file.")
            else:
                new_file_path = os.path.join(
                    DEFAULT_DATA_DIR, f"{new_file_name}.txt")
                if os.path.exists(new_file_path):
                    st.error(
                        f"A file named '{new_file_name}.txt' already exists.")
                else:
                    with open(new_file_path, "w", encoding="utf-8") as f:
                        f.write(new_file_content)
                    st.success(f"File '{new_file_name}.txt' has been created.")
                    update_db(DEFAULT_DATA_DIR, user_db_dir, chunk_size,
                              chunk_overlap, embedding_model, hf_token)
                    st.rerun()

    st.header("Current .txt Files")
    files = [f for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith('.txt')]

    if files:
        for file in files:
            file_path = os.path.join(DEFAULT_DATA_DIR, file)

            col1, col2, col3 = st.columns([4, 1, 1])

            with col1:
                st.write(file)

            with col2:
                confirm_delete = st.checkbox(
                    "Confirm Delete", key=f"confirm_{file}")

            with col3:
                if st.button("Delete", key=f"delete_{file}"):
                    if confirm_delete:
                        os.remove(file_path)
                        st.success(f"{file} has been deleted.")
                        update_db(DEFAULT_DATA_DIR, user_db_dir, chunk_size,
                                  chunk_overlap, embedding_model, hf_token)
                        st.rerun()
                    else:
                        st.warning(
                            "Please confirm deletion by checking the box.")
    else:
        st.write("No .txt files available.")

    settings.update({
        "embedding_model": embedding_model,
        "hf_token": hf_token,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "data_dir": DEFAULT_DATA_DIR,
        "db_dir": db_dir
    })
    save_settings(user_path, settings)
