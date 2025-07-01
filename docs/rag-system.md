# RAG（Retrieval-Augmented Generation）システムの仕組み

## 🎯 RAGとは

RAGは「検索拡張生成」の略で、外部の知識ベースから関連情報を検索し、その情報を基にLLMが回答を生成する手法です。

## 🤔 なぜRAGが必要なのか

### LLMの制限
- **知識の更新**: 学習データの時点で知識が固定
- **ハルシネーション**: 事実でない情報を生成する可能性
- **専門知識**: 特定ドメインの詳細な情報に限界

### RAGのメリット
- **最新情報**: リアルタイムで情報を更新可能
- **事実性向上**: 検索された情報に基づいた回答
- **コスト効率**: 全てをファインチューニングする必要がない

## 🔄 RAGシステムの流れ

### 1. 事前処理フェーズ
```
文書収集 → チャンク分割 → ベクトル化 → データベース保存
```

### 2. 検索・生成フェーズ
```
ユーザー質問 → ベクトル化 → 類似検索 → 関連文書取得 → LLM生成
```

## 🧩 主要コンポーネント

### 1. 文書分割（Chunking）
```python
def chunk_text(text, chunk_size=500, overlap=50):
    """テキストを重複ありでチャンクに分割"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
```

### 2. 埋め込みベクトル
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([
    "自然言語処理は面白い",
    "機械学習の基礎を学ぼう"
])
```

### 3. ベクトルデータベース

#### ChromaDB
- 軽量でセットアップが簡単
- Pythonネイティブ
- 小〜中規模データに適している

#### FAISS
- Facebook AI Research製
- 高速な類似検索
- 大規模データに対応

### 4. 検索戦略

#### セマンティック検索
```python
def semantic_search(query, vectordb, top_k=5):
    query_embedding = embed_model.encode([query])
    results = vectordb.similarity_search(query_embedding, k=top_k)
    return results
```

#### ハイブリッド検索
- セマンティック検索 + キーワード検索
- より精度の高い検索が可能

## 🏗️ 実装アーキテクチャ

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   PDFファイル   │ -> │  テキスト抽出  │ -> │  チャンク分割  │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  ベクトル検索  │ <- │ ベクトルDB保存 │ <- │  埋め込み生成  │
└─────────────┘    └─────────────┘    └─────────────┘
        │
        ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  ユーザー質問  │ -> │  関連文書取得  │ -> │   LLM生成   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 💡 実装のベストプラクティス

### 1. チャンクサイズの調整
- **小さすぎる**: 文脈が失われる
- **大きすぎる**: ノイズが多くなる
- **推奨**: 200-800文字、ドメインに応じて調整

### 2. 埋め込みモデルの選択
- **多言語対応**: multilingual-e5-large
- **日本語特化**: text-embedding-ada-002
- **軽量**: all-MiniLM-L6-v2

### 3. メタデータの活用
```python
chunks = [
    {
        "text": "チャンクの内容",
        "metadata": {
            "source": "document1.pdf",
            "page": 1,
            "timestamp": "2024-01-01"
        }
    }
]
```

## 🔧 実用的なテクニック

### 1. Re-ranking
検索結果を再順位付けして精度向上

### 2. Query Expansion
ユーザーの質問を拡張してより良い検索結果を取得

### 3. Context Compression
関連度の低い部分を圧縮してLLMの入力を最適化

## 📊 評価指標

- **検索精度**: 関連文書の取得率
- **回答品質**: 事実性、完全性、流暢性
- **レスポンス時間**: エンドツーエンドの処理時間

## 🚀 発展的なトピック

- **GraphRAG**: 知識グラフとRAGの組み合わせ
- **Agentic RAG**: AIエージェントによる動的な検索戦略
- **Multi-modal RAG**: テキスト以外の情報も活用
