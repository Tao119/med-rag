import streamlit as st
from utils.auth import login_user, register_user, logout_user, is_logged_in, get_logged_in_user, get_user_path
from pages_data.file_management import file_management_page
from pages_data.question_page import question_page
from pages_data.history_page import history_page


def main():
    st.set_page_config(page_title="RAG App", layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not is_logged_in():
        show_login_page()
    else:
        show_main_app()


def show_login_page():
    st.title("Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            success, msg = login_user(username, password)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab2:
        new_username = st.text_input("New Username", key="reg_user")
        new_password = st.text_input(
            "New Password", type="password", key="reg_pass")
        if st.button("Register"):
            success, msg = register_user(new_username, new_password)
            if success:
                st.success(msg)
                login_user(new_username, new_password)
                st.rerun()
            else:
                st.error(msg)


def show_main_app():
    username = get_logged_in_user()
    user_path = get_user_path(username)

    st.sidebar.title(f"Welcome, {username}")
    if st.sidebar.button("Logout"):
        logout_user()
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


if __name__ == "__main__":
    main()
