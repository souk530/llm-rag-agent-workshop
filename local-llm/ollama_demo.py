"""
ローカルLLMデモ：Ollamaを使った基本的な操作
"""

import subprocess
import json
import requests
import time

class OllamaDemo:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def check_ollama_status(self):
        """Ollamaサービスの状態を確認"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return True, "Ollamaサービスが実行中です"
            else:
                return False, f"Ollamaサービスエラー: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Ollamaサービスが起動していません"
    
    def list_models(self):
        """インストール済みモデルのリストを取得"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                return models.get('models', [])
            return []
        except Exception as e:
            print(f"モデルリスト取得エラー: {e}")
            return []
    
    def pull_model(self, model_name):
        """モデルをダウンロード"""
        print(f"モデル '{model_name}' をダウンロード中...")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'status' in data:
                        print(f"ステータス: {data['status']}")
                    if data.get('status') == 'success':
                        print(f"モデル '{model_name}' のダウンロード完了！")
                        return True
                        
        except Exception as e:
            print(f"ダウンロードエラー: {e}")
            return False
    
    def generate_text(self, model_name, prompt, stream=True):
        """テキスト生成"""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": stream
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream
            )
            
            if stream:
                full_response = ""
                print("生成中: ", end="", flush=True)
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            print(data['response'], end="", flush=True)
                            full_response += data['response']
                        if data.get('done', False):
                            print()  # 改行
                            break
                return full_response
            else:
                result = response.json()
                return result.get('response', '')
                
        except Exception as e:
            print(f"生成エラー: {e}")
            return ""
    
    def chat_demo(self, model_name):
        """対話デモ"""
        print(f"\n=== {model_name} との対話デモ ===")
        print("'quit' または 'exit' で終了")
        print("-" * 40)
        
        while True:
            user_input = input("\nあなた: ")
            if user_input.lower() in ['quit', 'exit', '終了']:
                print("対話を終了します。")
                break
            
            print(f"{model_name}: ", end="")
            response = self.generate_text(model_name, user_input)
            print()  # 改行

def setup_ollama():
    """Ollama初期セットアップ"""
    print("=== Ollama セットアップ ===")
    
    # Ollamaインストール確認
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Ollama インストール済み: {result.stdout.strip()}")
        else:
            print("Ollamaがインストールされていません。")
            print("https://ollama.ai からダウンロードしてください。")
            return False
    except FileNotFoundError:
        print("Ollamaが見つかりません。インストールしてください。")
        return False
    
    # Ollamaサービス起動
    print("Ollamaサービスを起動中...")
    try:
        subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        time.sleep(3)  # サービス起動待機
        print("Ollamaサービス起動完了")
        return True
    except Exception as e:
        print(f"サービス起動エラー: {e}")
        return False

def recommend_models():
    """推奨モデルの説明"""
    models = {
        "llama3.2": {
            "size": "約2GB",
            "description": "高性能な汎用モデル、日本語対応",
            "use_case": "一般的なタスク、対話"
        },
        "gemma2": {
            "size": "約1.6GB", 
            "description": "Googleが開発した軽量モデル",
            "use_case": "軽量な推論タスク"
        },
        "qwen2": {
            "size": "約1.9GB",
            "description": "中国語・英語に強いモデル",
            "use_case": "多言語対応が必要な場合"
        },
        "codellama": {
            "size": "約3.8GB",
            "description": "コード生成特化モデル",
            "use_case": "プログラミング支援"
        }
    }
    
    print("\n=== 推奨モデル ===")
    for name, info in models.items():
        print(f"\n{name}:")
        print(f"  サイズ: {info['size']}")
        print(f"  説明: {info['description']}")
        print(f"  用途: {info['use_case']}")

def demo_use_cases():
    """ローカルLLMの活用例デモ"""
    
    use_cases = {
        "プライベート文書の要約": {
            "prompt": "以下の会議議事録を要約してください：\n[社内機密情報を含む文書]",
            "benefit": "外部サービスを使わずに機密情報を処理"
        },
        "コード解説": {
            "prompt": "以下のPythonコードの動作を説明してください：\ndef fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "benefit": "オフライン環境でのプログラミング学習"
        },
        "創作支援": {
            "prompt": "宇宙を舞台にしたSF小説のプロット案を3つ提案してください。",
            "benefit": "著作権を気にせずアイデア出し"
        },
        "語学学習": {
            "prompt": "以下の英文を日本語に翻訳し、文法説明も加えてください：\n'The quick brown fox jumps over the lazy dog.'",
            "benefit": "ネット接続不要の個人学習"
        }
    }
    
    print("\n=== ローカルLLM活用例 ===")
    for title, case in use_cases.items():
        print(f"\n【{title}】")
        print(f"例: {case['prompt'][:50]}...")
        print(f"メリット: {case['benefit']}")

def security_comparison():
    """セキュリティ比較"""
    print("\n=== セキュリティ比較 ===")
    
    comparison = {
        "ローカルLLM": {
            "データ保存": "ローカルのみ",
            "通信": "不要", 
            "プライバシー": "高",
            "コスト": "初期設定のみ",
            "性能": "ハードウェア依存"
        },
        "クラウドLLM": {
            "データ保存": "クラウド",
            "通信": "必要",
            "プライバシー": "サービス依存",
            "コスト": "使用量課金",
            "性能": "高性能"
        }
    }
    
    for service, features in comparison.items():
        print(f"\n{service}:")
        for feature, value in features.items():
            print(f"  {feature}: {value}")

def main():
    """メイン実行関数"""
    print("ローカルLLM（Ollama）デモ")
    print("=" * 40)
    
    # 推奨モデル紹介
    recommend_models()
    
    # セキュリティ比較
    security_comparison()
    
    # 活用例
    demo_use_cases()
    
    # Ollamaセットアップ
    if not setup_ollama():
        print("セットアップに失敗しました。")
        return
    
    # デモ実行
    demo = OllamaDemo()
    
    # サービス状態確認
    status, message = demo.check_ollama_status()
    print(f"\nOllamaステータス: {message}")
    
    if not status:
        print("Ollamaサービスを手動で起動してください: ollama serve")
        return
    
    # インストール済みモデル確認
    models = demo.list_models()
    if models:
        print(f"\nインストール済みモデル: {len(models)}個")
        for model in models:
            print(f"  - {model['name']}")
    else:
        print("\nモデルがインストールされていません。")
        print("推奨: ollama pull llama3.2")
        
        # モデルダウンロードの提案
        response = input("llama3.2をダウンロードしますか？ (y/N): ")
        if response.lower() == 'y':
            demo.pull_model("llama3.2")
    
    # 対話デモ（モデルがある場合）
    if models:
        model_name = models[0]['name']
        print(f"\n{model_name} での対話デモを開始します")
        demo.chat_demo(model_name)

if __name__ == "__main__":
    main()
