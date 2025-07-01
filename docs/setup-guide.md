# 環境構築手順書

## 🖥️ システム要件

- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8以上（推奨: 3.9-3.11）
- **メモリ**: 最低4GB（推奨: 8GB以上）
- **ストレージ**: 2GB以上の空き容量

## 📋 事前準備チェックリスト

### 1. Pythonのインストール確認

コマンドプロンプト/ターミナルで以下を実行：

```bash
python --version
# または
python3 --version
```

**期待される出力例**: `Python 3.9.7`

❌ **エラーが出る場合**: [Python公式サイト](https://www.python.org/downloads/)からダウンロード・インストール

### 2. pipの確認

```bash
pip --version
# または  
pip3 --version
```

### 3. Gitのインストール確認

```bash
git --version
```

❌ **エラーが出る場合**: [Git公式サイト](https://git-scm.com/)からダウンロード・インストール

## 🚀 プロジェクトセットアップ

### Step 1: プロジェクトのクローン

```bash
git clone https://github.com/your-repo/llm-rag-agent-workshop.git
cd llm-rag-agent-workshop
```

### Step 2: 仮想環境の作成

#### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

✅ **正常な場合**: プロンプトの先頭に `(venv)` が表示される

### Step 3: 依存関係のインストール

```bash
pip install -r requirements.txt
```

⏱️ **所要時間**: 3-5分

### Step 4: 環境変数の設定

1. `.env.example`をコピーして`.env`を作成：

```bash
cp .env.example .env
```

2. `.env`ファイルを編集してAPIキーを設定：

```
GOOGLE_API_KEY=your_actual_api_key_here
```

## 🔑 APIキーの取得方法

### Google Gemini API

1. [Google AI Studio](https://aistudio.google.com/)にアクセス
2. Googleアカウントでログイン
3. 「Create API Key」をクリック
4. 生成されたキーをコピーして`.env`ファイルに貼り付け

### その他のAPIキー（オプション）

#### OpenWeatherMap API
- [OpenWeatherMap](https://openweathermap.org/api)で無料アカウント作成
- APIキーを取得して`WEATHER_API_KEY`に設定

#### News API
- [News API](https://newsapi.org/)で無料アカウント作成
- APIキーを取得して`NEWS_API_KEY`に設定

## ✅ 動作確認

### 1. 基本テスト

```bash
python -c "import google.generativeai as genai; print('✅ Google AI SDK導入成功')"
python -c "import chromadb; print('✅ ChromaDB導入成功')"
python -c "import sentence_transformers; print('✅ Sentence Transformers導入成功')"
```

### 2. API接続テスト

```bash
python demos/prompt_engineering_demo.py
```

✅ **期待される動作**: プロンプトエンジニアリングのデモが実行される

### 3. ローカルLLMテスト（オプション）

```bash
python local-llm/ollama_demo.py
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 問題1: `pip install`でエラーが発生
```
ERROR: Could not install packages due to an EnvironmentError
```

**解決方法**:
```bash
pip install --user -r requirements.txt
```

#### 問題2: Python版本が古い
```
Python version 3.7 is not supported
```

**解決方法**: Python 3.8以上をインストール

#### 問題3: モジュールが見つからない
```
ModuleNotFoundError: No module named 'google'
```

**解決方法**:
1. 仮想環境が有効化されているか確認
2. 依存関係を再インストール
```bash
pip install --upgrade -r requirements.txt
```

#### 問題4: APIキーエラー
```
google.api_core.exceptions.Unauthenticated: 401 API key not valid
```

**解決方法**:
1. `.env`ファイルの場所を確認
2. APIキーが正しく設定されているか確認
3. APIキーに不要な空白が含まれていないか確認

#### 問題5: メモリ不足
```
OutOfMemoryError
```

**解決方法**:
- 軽量な埋め込みモデルを使用
- バッチサイズを削減
- 不要なプロセスを終了

### 各OS固有の注意点

#### Windows
- 長いパス名でエラーが出る場合は、Git Bashの使用を推奨
- ウイルス対策ソフトがPythonファイルをブロックしていないか確認

#### macOS
- Xcode Command Line Toolsが必要な場合があります：
```bash
xcode-select --install
```

#### Linux
- システムパッケージが不足している場合：
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv
```

## 📱 推奨エディタ設定

### Visual Studio Code

推奨拡張機能：
- Python
- Pylance
- Python Docstring Generator
- GitLens

設定例（`.vscode/settings.json`）：
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

### PyCharm

1. プロジェクトを開く
2. File → Settings → Project → Python Interpreter
3. 作成した仮想環境を選択

## 🧪 ワークショップ前の最終確認

以下のコマンドで全ての動作確認を実行：

```bash
# デモが正常に動作するか確認
python demos/prompt_engineering_demo.py

# RAGシステムのテスト
python hands-on/option-a-rag/rag_system.py

# エージェントシステムのテスト
python hands-on/option-b-agent/agent_system.py
```

全て正常に動作すれば、ワークショップの準備完了です！

## 📞 サポート

当日、技術的な問題が発生した場合：

1. **講師に質問**: 遠慮なくお声がけください
2. **ペアプログラミング**: 隣の方と助け合って解決
3. **公式ドキュメント**: エラーメッセージをGoogle検索

## 🎯 ワークショップ当日の流れ

### 事前確認（9:00-9:30）
- [ ] 仮想環境の有効化確認
- [ ] APIキーの動作確認
- [ ] 必要なライブラリの import テスト

### 準備物
- ノートパソコン（充電器も忘れずに）
- メモ帳・筆記用具
- 飲み物・軽食
- 質問リスト

## 🌟 成功のコツ

1. **事前準備をしっかりと**: 当日は実習に集中できます
2. **積極的に質問**: 理解度を深めるチャンス
3. **ペアと協力**: 異なる視点から学べます
4. **手を動かす**: 実際にコードを書いて体験
5. **メモを取る**: 後で振り返れるように

頑張って素晴らしいワークショップにしましょう！🚀
