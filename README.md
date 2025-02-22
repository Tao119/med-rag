# team-d RAG test

## 概要

**team-d RAG test** は、医療特化型の **Retrieval-Augmented Generation (RAG)** システムです。  
アップロードされた医療関連ドキュメントをベクトル化し、**Azure OpenAI** を活用してユーザーの質問に対して最適な回答を生成します。  
**Streamlit** ベースの Web インターフェースを備えており、簡単な操作で使用可能です。

---

## 機能

- 📁 **ドキュメントアップロード:**  
  テキストファイル（.txt）のアップロードに対応し、知識ベースを拡張可能。

- 🔍 **RAG (Retrieval-Augmented Generation):**  
  ベクトルストアから関連ドキュメントを取得し、Azure OpenAI で質問に回答。

- ⚙️ **パラメータ調整:**

  - 取得するドキュメント数 (`k`)
  - スコアのしきい値
  - テキストのチャンクサイズ
  - Embedding モデルの指定

- 💾 **ベクトルストア管理:**  
  ドキュメントをベクトル化し、ローカルストレージに保存。

- 🧠 **Azure OpenAI GPT-4o との連携:**  
  高精度な回答生成機能を実装。

---

## ディレクトリ構成

```bash
.
├── app.py                  # Streamlit アプリケーション
├── embedding.py            # 埋め込みモデルのクラス
├── requirements.txt        # 必要なパッケージ
├── .env                    # 環境変数ファイル (API キーなど)
├── /data                   # アップロードされたドキュメント
└─── /db                     # ベクトルストアの保存先
```

---

## ⚡ セットアップ手順

### 1️⃣ **リポジトリをクローン**

```bash
git clone git@github.com:Tao119/med-rag.git
cd med-rag
```

### 2️⃣ **Python 仮想環境の作成**

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
```

### 3️⃣ **必要パッケージのインストール**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ **環境変数の設定**

プロジェクトルートに **`.env`** ファイルを作成し、以下を記述してください。

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# HuggingFace Embedding Model
HUGGINGFACE_EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
HUGGINGFACE_API_TOKEN=your_huggingface_api_token
```

### 5️⃣ **Streamlit アプリの起動**

```bash
streamlit run app.py
```

---

## 💡 使い方

1. **ドキュメントアップロード:**
   メイン画面から `.txt` ファイルをアップロードします。

2. **パラメータ設定:**
   サイドバーで以下の項目を設定できます。

   - 🔢 **取得するドキュメント数 (`k`)**
   - 🎯 **スコアのしきい値**
   - 📏 **チャンクサイズ**
   - 🔗 **Embedding モデル名**
   - 🔑 **HuggingFace API トークン**

3. **データベース更新:**
   「🔄 データベースを更新」をクリックして、新しいドキュメントをベクトル化します。

4. **質問を入力:**
   「💬 質問を入力してください」の欄に質問を記入し、「🚀 質問する」をクリック。

5. **回答と関連ドキュメント表示:**
   - **💡 回答:** Azure OpenAI GPT-4o による回答。
   - **📄 検索されたドキュメント:** 関連性の高いドキュメントとスコアが表示されます。

---
