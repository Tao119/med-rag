import json
import os


from dotenv import load_dotenv

load_dotenv()

USER_DATA_DIR = "users"
os.makedirs(USER_DATA_DIR, exist_ok=True)


def load_history(user_path):
    history_file = os.path.join(user_path, "history.json")

    if not os.path.exists(history_file):
        return []

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []


def load_global_history():
    global_history = []

    for username in os.listdir(USER_DATA_DIR):
        user_path = os.path.join(USER_DATA_DIR, username)
        history_file = os.path.join(user_path, "history.json")

        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    user_history = json.load(f)
                    if isinstance(user_history, list):
                        global_history.extend(user_history)
                    else:
                        print(f"{username} の history.json はリスト形式ではありません。")
            except json.JSONDecodeError:
                print(f"{username} の history.json にJSONDecodeErrorが発生しました。")
            except Exception as e:
                print(f"{username} の履歴ファイル読み込み中にエラーが発生しました: {e}")

    return global_history


def save_history(user_path, entry):
    history = load_history(user_path)
    history.append(entry)
    history_file = os.path.join(user_path, "history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
