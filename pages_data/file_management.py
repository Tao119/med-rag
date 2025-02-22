import streamlit as st
import os
from utils.settings_utils import load_settings, save_settings

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

from embedding import Embeddings
from tqdm import tqdm


DEFAULT_DATA_DIR = "./data"
DEFAULT_DB_DIR = "./db"


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
    # Load user-specific settings
    settings = load_settings(user_path)

    st.title("File Management")

    # Sidebar configuration for embeddings and chunking
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

    # Directories for data and DB
    st.sidebar.header("Data Directory Settings")
    data_dir = st.sidebar.text_input(
        "Data Directory", settings.get("data_dir", DEFAULT_DATA_DIR))
    db_dir = st.sidebar.text_input(
        "Database Directory", settings.get("db_dir", DEFAULT_DB_DIR))

    if st.sidebar.button("Save Paths"):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        settings["data_dir"] = data_dir
        settings["db_dir"] = db_dir
        save_settings(user_path, settings)
        update_db(data_dir, db_dir, chunk_size,
                  chunk_overlap, embedding_model, hf_token)
        st.success("Paths updated successfully.")
        st.rerun()

    # Tabs for Uploading and Creating Files
    tab1, tab2 = st.tabs(["Upload", "Create"])

    with tab1:
        st.header("Upload New File")
        uploaded_file = st.file_uploader("Upload .txt files", type=["txt"])

        if uploaded_file is not None:
            save_path = os.path.join(data_dir, uploaded_file.name)

            if os.path.exists(save_path):
                st.error(
                    f"A file named '{uploaded_file.name}' already exists. Please use a different name.")
            else:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"{uploaded_file.name} has been uploaded.")
                update_db(data_dir, db_dir, chunk_size,
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
                new_file_path = os.path.join(data_dir, f"{new_file_name}.txt")
                if os.path.exists(new_file_path):
                    st.error(
                        f"A file named '{new_file_name}.txt' already exists.")
                else:
                    with open(new_file_path, "w", encoding="utf-8") as f:
                        f.write(new_file_content)
                    st.success(f"File '{new_file_name}.txt' has been created.")
                    update_db(data_dir, db_dir, chunk_size,
                              chunk_overlap, embedding_model, hf_token)
                    st.rerun()

    # Displaying Current Files
    st.header("Current .txt Files")
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

    if files:
        for file in files:
            file_path = os.path.join(data_dir, file)

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
                        update_db(data_dir, db_dir, chunk_size,
                                  chunk_overlap, embedding_model, hf_token)
                        st.rerun()
                    else:
                        st.warning(
                            "Please confirm deletion by checking the box.")
    else:
        st.write("No .txt files available.")

    # Update settings at the end
    settings.update({
        "embedding_model": embedding_model,
        "hf_token": hf_token,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })
    save_settings(user_path, settings)
