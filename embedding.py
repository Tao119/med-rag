from transformers import AutoTokenizer, AutoModel, logging
import torch
import numpy as np
import os

# ログレベルを下げて警告を減らす
logging.set_verbosity_error()


class Embeddings:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", hf_token=None):
        # Hugging Face APIトークンで認証
        if hf_token:
            os.environ["HUGGINGFACE_API_TOKEN"] = hf_token

        # トークナイザーとモデルをロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=hf_token)
        self.model = AutoModel.from_pretrained(
            model_name, use_auth_token=hf_token)

    def _mean_pooling(self, model_output, attention_mask):
        """平均プーリングを使用して文埋め込みを取得"""
        token_embeddings = model_output.last_hidden_state  # 最終隠れ状態
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _normalize(self, vec):
        """L2正規化"""
        return vec / np.linalg.norm(vec)

    def embed_documents(self, texts):
        """複数のドキュメントを埋め込み"""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = self._mean_pooling(outputs, inputs['attention_mask'])
            norm_embedding = self._normalize(embedding.squeeze().numpy())
            embeddings.append(norm_embedding)
        return embeddings

    def embed_query(self, text):
        """単一のクエリを埋め込み"""
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = self._mean_pooling(outputs, inputs['attention_mask'])
        norm_embedding = self._normalize(embedding.squeeze().numpy())
        return norm_embedding
