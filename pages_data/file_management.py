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


def save_chunks_to_json(user_chunks_dir, file_name, chunks):
    """チャンクデータをJSONに保存"""
    os.makedirs(user_chunks_dir, exist_ok=True)
    chunks_data = [{
        "chunk_number": chunk.metadata["chunk_number"],
        "content": chunk.page_content
    } for chunk in chunks]

    json_path = os.path.join(user_chunks_dir, f"{file_name}_chunks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)


def load_chunks_from_json(user_chunks_dir, file_name):
    """保存されたチャンクデータを読み込む"""
    json_path = os.path.join(user_chunks_dir, f"{file_name}_chunks.json")
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_chunks_json(user_chunks_dir, file_name):
    """ファイル削除時にチャンクデータも削除"""
    json_path = os.path.join(user_chunks_dir, f"{file_name}_chunks.json")
    if os.path.exists(json_path):
        os.remove(json_path)


def vectorize_file(file_path, db_dir, user_chunks_dir, chunk_size, chunk_overlap, embedding_model, hf_token):
    """ファイルをベクトル化し、チャンクデータを保存"""
    loader = DirectoryLoader(os.path.dirname(
        file_path), glob=os.path.basename(file_path))
    documents = loader.load()

    if not documents:
        st.warning(f"No documents found in {file_path}.")
        return []
    file_db_dir = os.path.join(db_dir, file_path.split("/")[-1])

    chroma_db = Chroma(persist_directory=file_db_dir, embedding_function=Embeddings(
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
        persist_directory=file_db_dir,
    )

    save_chunks_to_json(
        user_chunks_dir, os.path.basename(file_path), all_chunks)

    st.success(
        f"✅ {os.path.basename(file_path)} vectorized into {len(all_chunks)} chunks with {chunk_size} size and {chunk_overlap}% overlap."
    )

    return all_chunks


def file_management_page(user_path):
    st.title("📁 File Management with Per-File Settings")

    user_db_dir = os.path.join(user_path, "db")
    user_chunks_dir = os.path.join(user_path, "chunks_data")
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    os.makedirs(user_db_dir, exist_ok=True)
    os.makedirs(user_chunks_dir, exist_ok=True)

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
                confirm_all = st.checkbox(
                    "Check to include vectorized", key=f"confirm_all")
                if st.button("🔄 Vectorize All Files"):
                    updated_files = []
                    for file in files:
                        if file in file_settings and not confirm_all:
                            continue

                        file_path = os.path.join(DEFAULT_DATA_DIR, file)
                        settings = {
                            "chunk_size": global_settings.get("chunk_size", 1024),
                            "chunk_overlap": global_settings.get("chunk_overlap", 20),
                            "embedding_model": global_settings.get("embedding_model", ""),
                            "hf_token": global_settings.get("hf_token", "")
                        }

                        vectorize_file(file_path, user_db_dir,
                                       user_chunks_dir, ** settings)
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

                display_files = []
                for f in files:
                    is_vectorized = f in file_settings
                    status_icon = "✅" if is_vectorized else "⚠️"
                    display_files.append(f"{status_icon} {f}")

                selected_file_display = st.radio(
                    "Select a file", display_files, key="file_selector")
                selected_file = selected_file_display.split(" ", 1)[1]

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

                # 個別ファイルのベクトル化
                col_v, col_d = st.columns([2, 1])

                with col_v:
                    if st.button(f"🚀 Vectorize {selected_file}", key=f"vectorize_{selected_file}"):
                        updated_settings = {
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "embedding_model": embedding_model,
                            "hf_token": hf_token
                        }
                        file_settings[selected_file] = updated_settings
                        save_file_settings(file_settings, user_path)
                        vectorize_file(file_path, user_db_dir,
                                       user_chunks_dir, **updated_settings)
                        st.success(
                            f"✅ {selected_file} has been vectorized with updated settings.")
                        st.rerun()

                with col_d:
                    if st.button(f"🗑️ Delete DB", key=f"delete_db_{selected_file}"):
                        db_dir = os.path.join(user_db_dir, selected_file)
                        if os.path.exists(db_dir):
                            import shutil
                            shutil.rmtree(db_dir)
                            st.success(
                                f"🗑️ Vector DB for {selected_file} has been deleted.")
                        else:
                            st.warning(
                                f"⚠️ No Vector DB found for {selected_file}.")

                        if selected_file in file_settings:
                            del file_settings[selected_file]
                            save_file_settings(file_settings, user_path)

                        st.rerun()

                # チャンクのロード
                chunks = load_chunks_from_json(user_chunks_dir, selected_file)
                if not chunks:
                    st.warning(
                        "⚠️ No chunk data found. Please vectorize the file.")

                # チャンク一覧表示
                st.subheader("📋 Chunks")
                chunk_titles = [
                    f"Chunk {i + 1}: {chunk['content'][:30]}..." for i, chunk in enumerate(chunks)]
                if chunk_titles:
                    selected_chunk = st.selectbox(
                        "Select a Chunk to View", chunk_titles)
                    selected_chunk_index = chunk_titles.index(selected_chunk)
                    st.write("### Chunk Content")
                    st.write(chunks[selected_chunk_index]["content"])
                else:
                    st.info("No chunks available. Please vectorize the file.")

                # --- ファイル削除機能 ---
                st.markdown("---")
                st.subheader("🗑️ Delete File")
                confirm_delete = st.checkbox(
                    "Check to confirm deletion", key=f"confirm_delete_{selected_file}")
                if st.button(f"❌ Delete {selected_file}", key=f"delete_{selected_file}") and confirm_delete:
                    os.remove(file_path)
                    delete_chunks_json(user_chunks_dir, selected_file)
                    if selected_file in file_settings:
                        del file_settings[selected_file]
                        save_file_settings(file_settings, user_path)
                    st.success(f"🗑️ {selected_file} has been deleted.")
                    st.rerun()

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
                    vectorize_file(save_path, user_db_dir, user_chunks_dir,
                                   chunk_size, chunk_overlap, embedding_model, hf_token)
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
                        vectorize_file(new_file_path, user_db_dir, user_chunks_dir,
                                       chunk_size, chunk_overlap, embedding_model, hf_token)
                        file_settings[f"{new_file_name}.txt"] = {
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "embedding_model": embedding_model,
                            "hf_token": hf_token
                        }
                        save_file_settings(file_settings, user_path)
                    st.rerun()
