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

    # 保存したチャンクを返却して、後で表示に使う
    return all_chunks


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

    # --- タブ構成 ---
    tab1, tab2, tab3 = st.tabs(
        ["📄 List Files", "📤 Upload File", "📝 Create File"])

    # --- 📄 ファイル一覧と詳細設定 ---
    with tab1:
        col1, col2 = st.columns([1, 3])

        # --- 📄 ファイル一覧と状態表示 (左カラム) ---
        with col1:
            st.header("📄 Files")
            files = [f for f in os.listdir(
                DEFAULT_DATA_DIR) if f.endswith('.txt')]

            if not files:
                st.write("No .txt files available.")
            else:
                # ファイル名の左にベクトル化状態アイコンを追加
                display_files = []
                for f in files:
                    is_vectorized = f in file_settings
                    status_icon = "✅" if is_vectorized else "⚠️"
                    display_files.append(f"{status_icon} {f}")

                selected_file_display = st.radio(
                    "Select a file", display_files, key="file_selector")

                # 選択されたファイル名からアイコンを削除して取得
                selected_file = selected_file_display.split(" ", 1)[1]

                # 一括ベクトル化ボタン
                if st.button("🔄 Vectorize All Files"):
                    updated_files = []
                    for file in files:
                        if file in file_settings:
                            continue

                        file_path = os.path.join(DEFAULT_DATA_DIR, file)
                        settings = {
                            "chunk_size": global_settings.get("chunk_size", 1024),
                            "chunk_overlap": global_settings.get("chunk_overlap", 20),
                            "embedding_model": global_settings.get("embedding_model", ""),
                            "hf_token": global_settings.get("hf_token", "")
                        }

                        vectorize_file(file_path, user_db_dir, **settings)
                        file_settings[file] = settings
                        updated_files.append(file)

                    if updated_files:
                        save_file_settings(file_settings, user_path)
                        st.success(
                            f"✅ Vectorized and updated settings for: {', '.join(updated_files)}")
                    else:
                        st.info(
                            "ℹ️ No files were updated (all have individual settings).")

                    st.rerun()

        # --- 📋 選択されたファイルの詳細設定 (右カラム) ---
        with col2:
            if selected_file:
                st.header(f"⚙️ Settings for {selected_file}")
                file_path = os.path.join(DEFAULT_DATA_DIR, selected_file)

                is_vectorized = selected_file in file_settings
                status_color = "green" if is_vectorized else "red"
                status_text = "✅ Vectorized" if is_vectorized else "⚠️ Not Vectorized"
                st.markdown(
                    f"<span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)

                settings = file_settings.get(selected_file, {
                    "chunk_size": global_settings.get("chunk_size", 1024),
                    "chunk_overlap": global_settings.get("chunk_overlap", 20),
                    "embedding_model": global_settings.get("embedding_model", ""),
                    "hf_token": global_settings.get("hf_token", "")
                })

                chunk_size = st.number_input(
                    "Chunk Size", min_value=256, max_value=4096, value=settings["chunk_size"], step=256, key=f"chunk_size_{selected_file}")
                chunk_overlap = st.slider(
                    "Chunk Overlap (%)", 0, 50, settings["chunk_overlap"], key=f"chunk_overlap_{selected_file}")
                embedding_model = st.text_input(
                    "Embedding Model", settings["embedding_model"], key=f"embedding_model_{selected_file}")
                hf_token = st.text_input(
                    "HuggingFace Token", settings["hf_token"], type="password", key=f"hf_token_{selected_file}")

                # 個別ファイルのベクトル化とチャンク取得
                if st.button(f"🚀 Vectorize {selected_file}", key=f"vectorize_{selected_file}"):
                    updated_settings = {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "embedding_model": embedding_model,
                        "hf_token": hf_token
                    }
                    file_settings[selected_file] = updated_settings
                    save_file_settings(file_settings, user_path)
                    chunks = vectorize_file(
                        file_path, user_db_dir, **updated_settings)

                    st.success(
                        f"✅ {selected_file} has been vectorized with updated settings.")
                    st.rerun()

                # --- 📋 チャンク一覧と内容表示 ---
                st.markdown("---")
                st.subheader("📋 View Chunks")

                # チャンクを表示
                chunks = vectorize_file(file_path, user_db_dir, **settings)
                chunk_titles = [
                    f"Chunk {i + 1}: {chunk.page_content[:30]}..." for i, chunk in enumerate(chunks)]

                selected_chunk = st.selectbox(
                    "Select a Chunk to View", chunk_titles)

                # 選択されたチャンクの詳細を表示
                selected_chunk_index = chunk_titles.index(selected_chunk)
                st.write("### Chunk Content")
                st.write(chunks[selected_chunk_index].page_content)

                # --- ファイル削除機能 ---
                st.markdown("---")
                st.subheader("🗑️ Delete File")
                confirm_delete = st.checkbox(
                    "Check to confirm deletion", key=f"confirm_delete_{selected_file}")
                if st.button(f"❌ Delete {selected_file}", key=f"delete_{selected_file}") and confirm_delete:
                    os.remove(file_path)
                    if selected_file in file_settings:
                        del file_settings[selected_file]
                        save_file_settings(file_settings, user_path)
                    st.success(f"🗑️ {selected_file} has been deleted.")
                    st.rerun()
