import os
import json
import uuid
import streamlit as st
from streamlit_javascript import st_javascript
from streamlit_cookies_manager import EncryptedCookieManager

# 初期化: EncryptedCookieManager を使用
cookies = EncryptedCookieManager(prefix="medai_", password="your-secret-key")
if not cookies.ready():
    st.stop()

# ユーザーデータを保存するディレクトリ
USER_DATA_DIR = "users"
os.makedirs(USER_DATA_DIR, exist_ok=True)


def get_user_path(username):
    return os.path.join(USER_DATA_DIR, username)


def generate_client_key():
    client_key_display = st.empty()
    if "client_key" not in st.session_state:
        st.session_state["client_key"] = None

    # Use Streamlit HTML component to run JavaScript
    st.components.v1.html("""
    <script>
        (function() {
            let clientKey = localStorage.getItem("client_key");
            if (!clientKey) {
                clientKey = self.crypto.randomUUID();
                localStorage.setItem("client_key", clientKey);
                console.log("Generated and Saved Client Key:", clientKey);
            } else {
                console.log("Existing Client Key:", clientKey);
            }

            // Post clientKey to Streamlit
            const streamlitIframe = window.parent;
            streamlitIframe.postMessage(
                { isClientKey: true, clientKey: clientKey },
                "*"
            );
        })();

        // Listen for Streamlit events
        window.addEventListener("message", (event) => {
            console.log("Message received from Streamlit:", event.data);
        });
    </script>
""", height=0)

    client_key = st.query_params.get(
        'client_key', [None])[0]

    if client_key:
        client_key_display.write(f"**Client Key:** {client_key}")
        st.success("Client Key successfully retrieved and displayed.")
    else:
        client_key_display.write("Waiting for Client Key...")

    return client_key


# クライアントキーを取得（必ず存在する）
def get_client_key():
    return generate_client_key()


# クッキーからユーザーデータリストを取得
def get_users_data():
    users_data_json = cookies.get("users_data")
    if users_data_json:
        return json.loads(users_data_json)
    return []


# ユーザーデータリストをクッキーに保存
def save_users_data(users_data):
    cookies["users_data"] = json.dumps(users_data)
    cookies.save()
    print("---------- cookies saved!! ----------")
    print(cookies)


# ユーザー登録
def register_user(username, password):
    user_path = get_user_path(username)
    if os.path.exists(user_path):
        return False, "Username already exists."

    os.makedirs(user_path, exist_ok=True)
    user_data = {"username": username, "password": password}

    with open(os.path.join(user_path, "user.json"), "w") as f:
        json.dump(user_data, f, indent=4)

    # クライアントキー取得（再生成しない）
    client_key = get_client_key()

    # ユーザーデータ取得・追加
    users_data = get_users_data()
    users_data.append({
        "username": username,
        "client_key": client_key,
        "logged_in": True
    })
    save_users_data(users_data)

    return True, "Registration successful."


# ログイン
def login_user(username, password):
    user_path = os.path.join(get_user_path(username), "user.json")
    if not os.path.exists(user_path):
        return False, "User does not exist."

    with open(user_path, "r") as f:
        user_data = json.load(f)

    if user_data["password"] == password:
        # クライアントキー取得（再生成しない）
        client_key = get_client_key()

        # ユーザーデータ取得・更新
        users_data = get_users_data()
        user_found = False
        for user in users_data:
            if user["username"] == username:
                user["client_key"] = client_key
                user["logged_in"] = True
                user_found = True
                break

        if not user_found:
            users_data.append({
                "username": username,
                "client_key": client_key,
                "logged_in": True
            })

        save_users_data(users_data)

        return True, "Login successful."
    else:
        return False, "Incorrect password."


# ログアウト
def logout_user():
    client_key = get_client_key()

    # ユーザーデータ取得・更新
    users_data = get_users_data()
    for user in users_data:
        if user["client_key"] == client_key:
            user["logged_in"] = False
            break

    save_users_data(users_data)

    # ローカルストレージからクライアントキー削除しない（永続化）


# ログインステータスを確認
def is_logged_in():
    client_key = get_client_key()
    users_data = get_users_data()

    for user in users_data:
        if user["client_key"] == client_key and user["logged_in"]:
            return True
    return False


# ログイン中のユーザー名を取得
def get_logged_in_user():
    client_key = get_client_key()
    users_data = get_users_data()

    for user in users_data:
        if user["client_key"] == client_key and user["logged_in"]:
            return user["username"]
    return None
