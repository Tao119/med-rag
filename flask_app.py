from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Constants
DEFAULT_DATA_DIR = "./data"
DEFAULT_DB_DIR = "./db"
HISTORY_FILE = "./history.json"
SETTINGS_FILE = "./settings.json"

# Ensure directories exist
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
os.makedirs(DEFAULT_DB_DIR, exist_ok=True)

# Load and Save Settings


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "embedding_model": os.getenv("HUGGINGFACE_EMBEDDING_MODEL_NAME", ""),
        "hf_token": os.getenv("HUGGINGFACE_API_TOKEN", ""),
        "chunk_size": 1000,
        "k": 4,
        "score_threshold": 0.75,
        "system_prompt": "あなたは優秀な医療アシスタントです。ユーザーの質問に対して、正確で信頼性の高い医療知識を基に分かりやすく回答してください。",
        "data_dir": DEFAULT_DATA_DIR,
        "db_dir": DEFAULT_DB_DIR
    }


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

# Load and Save History


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# Routes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/file-management')
def file_management():
    settings = load_settings()
    files = [f for f in os.listdir(settings["data_dir"]) if f.endswith('.txt')]
    return render_template('file_management.html', settings=settings, files=files)


@app.route('/upload-file', methods=['POST'])
def upload_file():
    settings = load_settings()
    file = request.files['file']
    if file and file.filename.endswith('.txt'):
        file.save(os.path.join(settings["data_dir"], file.filename))
        return jsonify({"message": f"{file.filename} uploaded successfully."}), 200
    return jsonify({"error": "Invalid file format. Only .txt files are allowed."}), 400


@app.route('/delete-file', methods=['POST'])
def delete_file():
    settings = load_settings()
    filename = request.json.get('filename')
    file_path = os.path.join(settings["data_dir"], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": f"{filename} deleted successfully."}), 200
    return jsonify({"error": "File not found."}), 404


@app.route('/question-page')
def question_page():
    settings = load_settings()
    return render_template('question_page.html', settings=settings)


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required."}), 400

    # Simulated Response
    response = f"**回答:** {query} に対するサンプル応答です。\n\n- ポイント1\n- ポイント2"
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response
    }
    save_history(history_entry)
    return jsonify({"response": response, "retrieved_docs": [{"title": "Doc1", "content": "関連文書の内容", "score": 0.95}]}), 200


@app.route('/history')
def history():
    return render_template('history.html', history=load_history())


@app.route('/update-settings', methods=['POST'])
def update_settings():
    new_settings = request.json
    save_settings(new_settings)
    return jsonify({"message": "Settings updated successfully."}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
