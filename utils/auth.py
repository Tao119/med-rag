import os
import json
import streamlit as st

USER_DATA_DIR = "users"
os.makedirs(USER_DATA_DIR, exist_ok=True)


def get_user_path(username):
    return os.path.join(USER_DATA_DIR, username)


def register_user(username, password):
    user_path = get_user_path(username)
    if os.path.exists(user_path):
        return False, "Username already exists."

    os.makedirs(user_path, exist_ok=True)
    user_data = {"username": username, "password": password}

    with open(os.path.join(user_path, "user.json"), "w") as f:
        json.dump(user_data, f, indent=4)

    return True, "Registration successful."


def login_user(username, password):
    user_path = os.path.join(get_user_path(username), "user.json")
    if not os.path.exists(user_path):
        return False, "User does not exist."

    with open(user_path, "r") as f:
        user_data = json.load(f)

    if user_data["password"] == password:
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        return True, "Login successful."
    else:
        return False, "Incorrect password."


def logout_user():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None


def is_logged_in():
    return st.session_state.get("logged_in", False)


def get_logged_in_user():
    return st.session_state.get("username", "")
