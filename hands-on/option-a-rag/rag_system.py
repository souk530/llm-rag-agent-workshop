"""
Option A: RAGシステムの構築
PDFやWebページからの情報抽出と質問応答システム
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import PyPDF2
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict
import time

# 環境変数読み込み
load_dotenv()

class RAGSystem:
    def __init__(self, collection_name="workshop_docs"):
        """RAGシステムの初期化"""
        
        # Google Gemini API設定
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.llm = genai.GenerativeModel('gemini-pro')
        
        # 埋め込みモデル
        print("埋め込みモデルを読み込み中...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ChromaDBクライアント
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print("RAGシステムの初期化完了")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDFからテキストを抽出"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"PDF読み込みエラー: {e}")
            return ""
    
    def extract_text_from_url(self, url: str) -> str:
        """Webページからテキストを抽出"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 不要な要素を削除
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # テキスト抽出
            text = soup.get_text()
            
            # クリーニング
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Webページ読み込みエラー: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """テキストをチャンクに分割"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # 文の境界で切る
            if end < text_length:
                # 最後の句点または改行を探す
                last_period = text.rfind('。', start, end)
                last_newline = text.rfind('\n', start, end)
                
                if last_period > start:
                    end = last_period + 1
                elif last_newline > start:
                    end = last_newline
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """文書をベクトルDBに追加"""
        print(f"{len(documents)}個の文書を処理中...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, doc in enumerate(documents):
            content = doc['content']
            source = doc.get('source', f'document_{i}')
            doc_type = doc.get('type', 'unknown')
            
            # チャンク分割
            chunks = self.chunk_text(content)
            print(f"{source}: {len(chunks)}チャンク")
            
            for j, chunk in enumerate(chunks):
                chunk_id = f"{source}_chunk_{j}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": source,
                    "type": doc_type,
                    "chunk_index": j,
                    "char_count": len(chunk)
                })
                all_ids.append(chunk_id)
        
        if not all_chunks:
            print("追加する文書がありません")
            return
        
        # 埋め込み生成
        print("埋め込みベクトルを生成中...")
        embeddings = self.embedding_model.encode(all_chunks).tolist()
        
        # ChromaDBに追加
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
            embeddings=embeddings
        )
        
        print(f"✅ {len(all_chunks)}個のチャンクをベクトルDBに追加完了")
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """関連するチャンクを検索"""
        # クエリの埋め込み生成
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # 検索実行
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # 結果を整理
        relevant_chunks = []
        for i in range(len(results['documents'][0])):
            relevant_chunks.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return relevant_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """コンテキストを基に回答を生成"""
        
        # コンテキストを構築
        context = "\n\n".join([
            f"【出典: {chunk['metadata']['source']}】\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        # プロンプト構築
        prompt = f"""
以下のコンテキスト情報を基に、質問に回答してください。

コンテキスト:
{context}

質問: {query}

回答の際は以下の点に注意してください：
1. コンテキストに含まれる情報のみを使用
2. 情報源を明記
3. わからない場合は「提供された情報では回答できません」と回答
4. 簡潔で分かりやすい回答を心がける

回答:
"""
        
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"回答生成エラー: {e}"
    
    def query(self, question: str, top_k: int = 5, show_sources: bool = True) -> Dict:
        """質問応答の実行"""
        print(f"\n質問: {question}")
        print("関連情報を検索中...")
        
        # 関連チャンク検索
        relevant_chunks = self.search_relevant_chunks(question, top_k)
        
        if not relevant_chunks:
            return {
                "answer": "関連する情報が見つかりませんでした。",
                "sources": []
            }
        
        # 回答生成
        print("回答を生成中...")
        answer = self.generate_answer(question, relevant_chunks)
        
        # ソース情報
        sources = []
        if show_sources:
            print("\n参照した情報源:")
            for i, chunk in enumerate(relevant_chunks, 1):
                source_info = f"[{i}] {chunk['metadata']['source']} (類似度: {1-chunk['distance']:.3f})"
                print(source_info)
                sources.append({
                    "source": chunk['metadata']['source'],
                    "content": chunk['content'][:200] + "...",
                    "similarity": 1 - chunk['distance']
                })
        
        return {
            "answer": answer,
            "sources": sources,
            "relevant_chunks": relevant_chunks
        }
    
    def get_collection_stats(self) -> Dict:
        """コレクションの統計情報"""
        count = self.collection.count()
        
        if count > 0:
            # メタデータから統計を取得
            all_data = self.collection.get()
            sources = set()
            doc_types = {}
            
            for metadata in all_data['metadatas']:
                sources.add(metadata['source'])
                doc_type = metadata.get('type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            return {
                "total_chunks": count,
                "unique_sources": len(sources),
                "document_types": doc_types,
                "sources": list(sources)
            }
        
        return {"total_chunks": 0, "unique_sources": 0, "document_types": {}, "sources": []}

def demo_with_sample_data():
    """サンプルデータでのデモ"""
    
    # RAGシステム初期化
    rag = RAGSystem("demo_collection")
    
    # サンプル文書
    sample_documents = [
        {
            "content": """
人工知能（AI）は、コンピューターシステムが人間のような知的行動を示す技術分野です。
機械学習、自然言語処理、コンピュータビジョンなどの技術を組み合わせて、複雑な問題を解決します。
近年、深層学習の発展により、AIの性能は飛躍的に向上しています。

AIの主要な応用分野：
1. 画像認識：医療診断、自動運転車の視覚システム
2. 自然言語処理：機械翻訳、チャットボット、文書要約
3. 推薦システム：ECサイト、動画配信サービス
4. 音声認識：スマートスピーカー、音声アシスタント

AIの課題：
- 説明可能性：AIの判断根拠が不明確
- バイアス：学習データの偏りが結果に影響
- プライバシー：個人情報の取り扱い
- 雇用への影響：自動化による職の変化
            """,
            "source": "AI基礎資料",
            "type": "教材"
        },
        {
            "content": """
機械学習は、データからパターンを学習してタスクを実行するAIの手法です。

主要な機械学習の種類：

1. 教師あり学習（Supervised Learning）
   - 正解データを使って学習
   - 分類、回帰問題
   - 例：スパムメール判定、株価予測

2. 教師なし学習（Unsupervised Learning）
   - 正解なしでデータの構造を発見
   - クラスタリング、次元削減
   - 例：顧客セグメンテーション、異常検知

3. 強化学習（Reinforcement Learning）
   - 試行錯誤を通じて最適な行動を学習
   - 報酬を最大化する方策を習得
   - 例：ゲームAI、ロボット制御

深層学習は機械学習の一分野で、ニューラルネットワークを多層化した手法です。
画像認識、自然言語処理で特に優れた性能を発揮します。
            """,
            "source": "機械学習入門",
            "type": "教材"
        },
        {
            "content": """
自然言語処理（NLP）は、人間の言語をコンピューターが理解・処理する技術です。

NLPの主要タスク：

1. 形態素解析：文を単語に分割
2. 構文解析：文の文法構造を解析
3. 意味解析：文の意味を理解
4. 機械翻訳：言語間の翻訳
5. 感情分析：テキストの感情を判定
6. 質問応答：質問に対する回答生成
7. 文書要約：長い文書の要約作成

最近の発展：
- Transformer：Attention機構による高性能モデル
- BERT：双方向エンコーダー
- GPT：生成型事前学習モデル
- ChatGPT：対話型AI

応用例：
- 検索エンジン
- 音声アシスタント
- 自動翻訳
- チャットボット
- 文書分類
            """,
            "source": "NLP技術概要",
            "type": "技術文書"
        }
    ]
    
    # 文書をRAGシステムに追加
    rag.add_documents(sample_documents)
    
    # 統計情報表示
    stats = rag.get_collection_stats()
    print(f"\n📊 コレクション統計:")
    print(f"総チャンク数: {stats['total_chunks']}")
    print(f"文書数: {stats['unique_sources']}")
    print(f"文書タイプ: {stats['document_types']}")
    
    # サンプル質問でテスト
    sample_questions = [
        "AIの主要な応用分野は何ですか？",
        "教師あり学習とは何ですか？",
        "自然言語処理の最近の発展について教えてください",
        "機械学習の種類を教えてください",
        "Transformerとは何ですか？"
    ]
    
    print("\n🤖 質問応答デモ")
    print("=" * 50)
    
    for question in sample_questions:
        result = rag.query(question)
        print(f"\n💬 {question}")
        print(f"🤖 {result['answer']}")
        print("-" * 50)

def interactive_demo():
    """インタラクティブなデモ"""
    
    print("=== RAGシステム インタラクティブデモ ===")
    rag = RAGSystem("interactive_collection")
    
    while True:
        print("\n選択してください:")
        print("1. 文書を追加 (テキスト入力)")
        print("2. PDFファイルを追加")
        print("3. Webページを追加")
        print("4. 質問する")
        print("5. 統計情報を表示")
        print("6. 終了")
        
        choice = input("\n選択 (1-6): ").strip()
        
        if choice == "1":
            print("\n文書の内容を入力してください (空行で終了):")
            content_lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                content_lines.append(line)
            
            if content_lines:
                content = "\n".join(content_lines)
                source = input("文書名を入力: ").strip() or "手入力文書"
                
                documents = [{
                    "content": content,
                    "source": source,
                    "type": "手入力"
                }]
                rag.add_documents(documents)
        
        elif choice == "2":
            pdf_path = input("PDFファイルのパスを入力: ").strip()
            if os.path.exists(pdf_path):
                content = rag.extract_text_from_pdf(pdf_path)
                if content:
                    documents = [{
                        "content": content,
                        "source": os.path.basename(pdf_path),
                        "type": "PDF"
                    }]
                    rag.add_documents(documents)
                else:
                    print("PDFの読み込みに失敗しました")
            else:
                print("ファイルが見つかりません")
        
        elif choice == "3":
            url = input("WebページのURLを入力: ").strip()
            if url.startswith("http"):
                content = rag.extract_text_from_url(url)
                if content:
                    documents = [{
                        "content": content,
                        "source": url,
                        "type": "Webページ"
                    }]
                    rag.add_documents(documents)
                else:
                    print("Webページの読み込みに失敗しました")
            else:
                print("有効なURLを入力してください")
        
        elif choice == "4":
            stats = rag.get_collection_stats()
            if stats['total_chunks'] == 0:
                print("まず文書を追加してください")
                continue
                
            question = input("\n質問を入力: ").strip()
            if question:
                result = rag.query(question)
                print(f"\n回答: {result['answer']}")
        
        elif choice == "5":
            stats = rag.get_collection_stats()
            print(f"\n📊 統計情報:")
            print(f"総チャンク数: {stats['total_chunks']}")
            print(f"文書数: {stats['unique_sources']}")
            print(f"文書タイプ: {stats['document_types']}")
            if stats['sources']:
                print("文書一覧:")
                for source in stats['sources']:
                    print(f"  - {source}")
        
        elif choice == "6":
            print("デモを終了します")
            break
        
        else:
            print("無効な選択です")

def batch_processing_demo():
    """バッチ処理デモ"""
    
    print("=== バッチ処理デモ ===")
    
    # サンプルデータディレクトリ作成
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # サンプルファイル作成
    sample_files = {
        "ai_history.txt": """
人工知能の歴史

1950年代：
- アラン・チューリングがチューリングテストを提案
- 「機械は考えることができるか？」という問いを提起

1960年代：
- 初期のエキスパートシステム開発
- ELIZA：初期の自然言語処理プログラム

1980年代：
- エキスパートシステムの商業化
- 第二次AIブーム

1990年代：
- 機械学習手法の発展
- ニューラルネットワークの再注目

2000年代：
- 深層学習の基礎研究
- ビッグデータの普及

2010年代：
- 深層学習ブレイクスルー
- ImageNet、AlphaGoの成功

2020年代：
- 大規模言語モデル（LLM）の登場
- GPT、BERT、ChatGPTの普及
        """,
        
        "ml_algorithms.txt": """
機械学習アルゴリズム一覧

線形回帰（Linear Regression）：
- 連続値予測の基本手法
- 解釈しやすい
- 線形関係を仮定

決定木（Decision Tree）：
- 分類・回帰両方に対応
- ルールベースで解釈しやすい
- 過学習しやすい

ランダムフォレスト（Random Forest）：
- 決定木のアンサンブル手法
- 高い精度
- 特徴量重要度がわかる

サポートベクトルマシン（SVM）：
- 高次元データに強い
- カーネル関数で非線形分離
- 計算コストが高い

k近傍法（k-NN）：
- シンプルな分類・回帰手法
- 局所的パターンを捉える
- 計算量が多い

ナイーブベイズ（Naive Bayes）：
- 確率的分類手法
- テキスト分類で有効
- 特徴量独立性を仮定

ニューラルネットワーク：
- 脳の神経細胞を模倣
- 非線形関係を学習
- 深層学習の基礎
        """
    }
    
    # ファイル作成
    for filename, content in sample_files.items():
        with open(data_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"サンプルデータを {data_dir} に作成しました")
    
    # RAGシステムで処理
    rag = RAGSystem("batch_collection")
    
    documents = []
    for file_path in data_dir.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents.append({
            "content": content,
            "source": file_path.name,
            "type": "テキストファイル"
        })
    
    rag.add_documents(documents)
    
    # テスト質問
    test_questions = [
        "人工知能の歴史における重要な出来事を教えてください",
        "1950年代のAI研究について教えてください",
        "ランダムフォレストの特徴は何ですか？",
        "テキスト分類に適したアルゴリズムはありますか？"
    ]
    
    print("\n📝 バッチ処理結果:")
    for question in test_questions:
        result = rag.query(question, show_sources=False)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")

def main():
    """メイン実行関数"""
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEYが設定されていません")
        print(".envファイルでAPIキーを設定してください")
        return
    
    print("🚀 RAGシステム ワークショップ")
    print("=" * 40)
    
    while True:
        print("\nデモを選択してください:")
        print("1. サンプルデータでのデモ")
        print("2. インタラクティブデモ")
        print("3. バッチ処理デモ")
        print("4. 終了")
        
        choice = input("\n選択 (1-4): ").strip()
        
        if choice == "1":
            demo_with_sample_data()
        elif choice == "2":
            interactive_demo()
        elif choice == "3":
            batch_processing_demo()
        elif choice == "4":
            print("ワークショップを終了します")
            break
        else:
            print("無効な選択です")

if __name__ == "__main__":
    main()
