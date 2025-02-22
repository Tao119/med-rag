import os
import json
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

cookies = EncryptedCookieManager(prefix="medai_", password="your-secret-key")
if not cookies.ready():
    st.stop()

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

    cookies["logged_in"] = "true"
    cookies["username"] = username
    cookies.save()

    return True, "Registration successful."


def login_user(username, password):
    user_path = os.path.join(get_user_path(username), "user.json")
    if not os.path.exists(user_path):
        return False, "User does not exist."

    with open(user_path, "r") as f:
        user_data = json.load(f)

    if user_data["password"] == password:
        cookies["logged_in"] = "true"
        cookies["username"] = username
        cookies.save()
        return True, "Login successful."
    else:
        return False, "Incorrect password."


def logout_user():
    cookies["logged_in"] = "false"
    cookies["username"] = ""
    cookies.save()


def is_logged_in():
    return cookies.get("logged_in") == "true"


def get_logged_in_user():
    return cookies.get("username", "")
