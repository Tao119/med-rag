import streamlit as st
import os
from utils.auth import login_user, register_user, logout_user, is_logged_in, get_logged_in_user, get_user_path
from pages_data.file_management import file_management_page, update_db
from pages_data.question_page import question_page
from pages_data.history_page import history_page
from utils.settings_utils import load_settings
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG App", layout="wide")


def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = is_logged_in()

    if not st.session_state["logged_in"]:
        show_login_page()
    else:
        show_main_app()


def show_login_page():
    st.title("Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # Login Tab
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            success, msg = login_user(username, password)
            if success:
                st.success(msg)
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                initialize_user_environment(username)
                st.rerun()  # Rerun to reflect login
            else:
                st.error(msg)

    # Register Tab
    with tab2:
        new_username = st.text_input("New Username", key="reg_user")
        new_password = st.text_input(
            "New Password", type="password", key="reg_pass")
        if st.button("Register"):
            success, msg = register_user(new_username, new_password)
            if success:
                st.success(msg)
                login_user(new_username, new_password)
                st.session_state["logged_in"] = True
                st.session_state["username"] = new_username
                initialize_user_environment(new_username)
                st.rerun()  # Rerun to reflect registration and login
            else:
                st.error(msg)


def show_main_app():
    username = get_logged_in_user()
    if not username:
        st.error("Session expired. Please log in again.")
        logout_user()
        st.rerun()

    user_path = get_user_path(username)

    st.sidebar.title(f"Welcome, {username}")
    if st.sidebar.button("Logout"):
        logout_user()
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.rerun()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["RAG Question Page", "File Management", "Query History"])

    if page == "File Management":
        file_management_page(user_path)
    elif page == "RAG Question Page":
        question_page(user_path, username)
    elif page == "Query History":
        history_page(user_path)


def initialize_user_environment(username):
    user_path = get_user_path(username)
    user_db_dir = os.path.join(user_path, "db")
    data_dir = "./data"

    settings = load_settings(user_path)

    if not os.path.exists(user_db_dir):
        os.makedirs(user_db_dir)
        st.info(f"Database directory created for user: {username}")

        embedding_model = settings.get("embedding_model", "default-model")
        hf_token = settings.get("hf_token", "")
        chunk_size = settings.get("chunk_size", 512)
        chunk_overlap = settings.get("chunk_overlap", 25)

        update_db(
            data_dir=data_dir,
            db_dir=user_db_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            hf_token=hf_token
        )

        st.success("Database initialized successfully!")


if __name__ == "__main__":
    main()
