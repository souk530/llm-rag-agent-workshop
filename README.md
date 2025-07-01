# 大学生向けLLM・RAG・AIエージェントワークショップ

## 📚 ワークショップ概要

このワークショップでは、最新のLLM（Large Language Model）技術、RAG（Retrieval-Augmented Generation）、AIエージェントについて学び、実際に手を動かして体験できます。

## 🕐 スケジュール

- **9:00-9:30** イントロダクション
- **9:30-10:30** LLMの基礎理論
- **10:30-11:30** 最新のLLM活用技術
- **11:30-12:00** 開発ツールのデモ
- **12:00-13:00** 昼休憩
- **13:00-14:30** ローカルLLMハンズオン
- **14:30-16:30** Gemini APIを使った開発実習
- **16:30-17:00** 成果発表とまとめ

## 🛠️ 事前準備

### 必要なソフトウェア
- Python 3.8以上
- Git
- VSCode（推奨）

### アカウント作成
- [Google AI Studio](https://aistudio.google.com/)でGemini API利用のためのアカウント作成

### 環境構築

1. このリポジトリをクローン
```bash
git clone https://github.com/your-repo/llm-rag-agent-workshop.git
cd llm-rag-agent-workshop
```

2. 仮想環境の作成と有効化
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

4. 環境変数の設定
```bash
cp .env.example .env
# .envファイルを編集してAPIキーを設定
```

## 📁 ディレクトリ構成

```
llm-rag-agent-workshop/
├── README.md                 # このファイル
├── requirements.txt          # 必要なライブラリ
├── .env.example             # 環境変数テンプレート
├── docs/                    # ドキュメント・資料
├── demos/                   # デモコード
├── hands-on/               # ハンズオン実習
│   ├── option-a-rag/       # Option A: RAGシステム
│   └── option-b-agent/     # Option B: AIエージェント
├── local-llm/              # ローカルLLM関連
└── resources/              # 学習リソース・サンプルデータ
```

## 🚀 実習内容

### Option A: RAGシステムの構築
- PDFやWebページからの情報抽出
- ChromaDBやFAISSを使ったベクトル検索
- 質問応答システムの実装

### Option B: AIエージェントの開発
- Function Callingの実装
- 外部APIとの連携
- タスク自動化エージェント

## 📖 学習リソース

- [Transformerアーキテクチャの解説](docs/transformer-architecture.md)
- [RAGシステムの仕組み](docs/rag-system.md)
- [AIエージェントの基礎](docs/ai-agents.md)

## 🤝 サポート

ワークショップ中に困ったことがあれば、お気軽にお声がけください！

## 📝 ライセンス

MIT License
