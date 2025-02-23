import streamlit as st
import os
import json
from utils.settings_utils import load_settings, save_settings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from embedding import Embeddings
from tqdm import tqdm

DEFAULT_DATA_DIR = "./data"


def load_file_settings(user_path):
    settings_file = os.path.join(user_path, "settings.json")
    if not os.path.exists(settings_file):
        return {}

    with open(settings_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_file_settings(settings, user_path):
    settings_file = os.path.join(user_path, "settings.json")
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=4)


def vectorize_file(file_path, db_dir, chunk_size, chunk_overlap, embedding_model, hf_token):
    loader = DirectoryLoader(os.path.dirname(
        file_path), glob=os.path.basename(file_path))
    documents = loader.load()

    if not documents:
        st.warning(f"No documents found in {file_path}.")
        return

    chroma_db = Chroma(persist_directory=db_dir, embedding_function=Embeddings(
        model_name=embedding_model, hf_token=hf_token))
    chroma_db.delete_collection()

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * (chunk_overlap / 100))
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source"] = os.path.basename(file_path)
            chunk.metadata["chunk_number"] = f"{idx + 1}/{len(chunks)}"
        all_chunks.extend(chunks)

    Chroma.from_documents(
        tqdm(all_chunks),
        Embeddings(model_name=embedding_model, hf_token=hf_token),
        persist_directory=db_dir,
    )

    st.success(
        f"✅ {os.path.basename(file_path)} vectorized with {chunk_size} chunk size and {chunk_overlap}% overlap.")


def file_management_page(user_path):
    st.title("📁 File Management with Per-File Settings")

    user_db_dir = os.path.join(user_path, "db")
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    os.makedirs(user_db_dir, exist_ok=True)

    global_settings = load_settings(user_path)
    file_settings = load_file_settings(user_path)

    # --- サイドバーの全体設定 ---
    st.sidebar.header("⚙️ Global Settings")
    embedding_model = st.sidebar.text_input(
        "Embedding model", global_settings.get("embedding_model", ""))
    hf_token = st.sidebar.text_input(
        "HuggingFace API Token", global_settings.get("hf_token", ""), type="password")
    chunk_size = st.sidebar.number_input(
        "Default Chunk Size", min_value=256, max_value=4096, value=global_settings.get("chunk_size", 1024), step=256)
    chunk_overlap = st.sidebar.slider(
        "Default Chunk Overlap (%)", 0, 50, global_settings.get("chunk_overlap", 20))

    # グローバル設定の保存
    if st.sidebar.button("💾 Save Global Settings"):
        global_settings.update({
            "embedding_model": embedding_model,
            "hf_token": hf_token,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        })
        save_settings(user_path, global_settings)
        st.success("✅ Global settings saved successfully.")

    # --- タブで表示を切り替え ---
    tab1, tab2, tab3 = st.tabs(
        ["📄 File List", "📤 Upload File", "📝 Create New File"])

    # --- 📄 ファイル一覧と設定 ---
    with tab1:
        st.header("📄 Files and Settings")
        files = [f for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith('.txt')]

        if files:
            for file in files:
                file_path = os.path.join(DEFAULT_DATA_DIR, file)
                st.subheader(file)

                # ベクトル化の状態表示
                is_vectorized = file in file_settings
                status_color = "green" if is_vectorized else "red"
                status_text = "✅ Vectorized" if is_vectorized else "⚠️ Not Vectorized"
                st.markdown(
                    f"<span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)

                # 詳細設定をexpanderで表示
                with st.expander("⚙️ Detailed Settings", expanded=False):
                    settings = file_settings.get(file, {
                        "chunk_size": global_settings.get("chunk_size", 1024),
                        "chunk_overlap": global_settings.get("chunk_overlap", 20),
                        "embedding_model": global_settings.get("embedding_model", ""),
                        "hf_token": global_settings.get("hf_token", "")
                    })

                    chunk_size = st.number_input(
                        f"Chunk Size ({file})", min_value=256, max_value=4096, value=settings["chunk_size"], step=256, key=f"chunk_size_{file}")
                    chunk_overlap = st.slider(
                        f"Chunk Overlap (%) ({file})", 0, 50, settings["chunk_overlap"], key=f"chunk_overlap_{file}")
                    embedding_model = st.text_input(
                        f"Embedding Model ({file})", settings["embedding_model"], key=f"embedding_model_{file}")
                    hf_token = st.text_input(
                        f"HuggingFace Token ({file})", settings["hf_token"], type="password", key=f"hf_token_{file}")

                    # 個別更新ボタン
                    if st.button(f"🚀 Vectorize {file}", key=f"vectorize_{file}"):
                        updated_settings = {
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "embedding_model": embedding_model,
                            "hf_token": hf_token
                        }
                        file_settings[file] = updated_settings
                        save_file_settings(file_settings, user_path)

                        vectorize_file(file_path, user_db_dir,
                                       **updated_settings)
                        st.success(
                            f"✅ {file} has been vectorized with updated settings.")

                st.markdown("---")
        else:
            st.write("No .txt files available.")

    # --- 📤 ファイルアップロード機能 ---
    with tab2:
        st.header("📤 Upload New File")
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        auto_vectorize = st.checkbox(
            "Automatically vectorize after upload", value=True)

        if uploaded_file is not None:
            save_path = os.path.join(DEFAULT_DATA_DIR, uploaded_file.name)

            if os.path.exists(save_path):
                st.error(
                    f"A file named '{uploaded_file.name}' already exists.")
            else:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ '{uploaded_file.name}' uploaded successfully.")

                if auto_vectorize:
                    # Use global settings for vectorization
                    vectorize_file(save_path, user_db_dir, chunk_size,
                                   chunk_overlap, embedding_model, hf_token)
                    # Save the settings for the new file
                    file_settings[uploaded_file.name] = {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "embedding_model": embedding_model,
                        "hf_token": hf_token
                    }
                    save_file_settings(file_settings, user_path)
                st.rerun()

    # --- 📝 新規テキストファイル作成 ---
    with tab3:
        st.header("📝 Create New Text File")
        new_file_name = st.text_input(
            "Enter new file name (without .txt extension)")
        new_file_content = st.text_area("Enter file content")
        create_auto_vectorize = st.checkbox(
            "Automatically vectorize after creation", value=True)

        if st.button("➕ Create File"):
            if not new_file_name:
                st.error("⚠️ Please enter a file name.")
            elif not new_file_content:
                st.error("⚠️ Please enter some content for the file.")
            else:
                new_file_path = os.path.join(
                    DEFAULT_DATA_DIR, f"{new_file_name}.txt")
                if os.path.exists(new_file_path):
                    st.error(
                        f"⚠️ A file named '{new_file_name}.txt' already exists.")
                else:
                    with open(new_file_path, "w", encoding="utf-8") as f:
                        f.write(new_file_content)
                    st.success(
                        f"✅ File '{new_file_name}.txt' created successfully.")

                    if create_auto_vectorize:
                        vectorize_file(
                            new_file_path, user_db_dir, chunk_size, chunk_overlap, embedding_model, hf_token)
                        file_settings[f"{new_file_name}.txt"] = {
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "embedding_model": embedding_model,
                            "hf_token": hf_token
                        }
                        save_file_settings(file_settings, user_path)
                    st.rerun()
