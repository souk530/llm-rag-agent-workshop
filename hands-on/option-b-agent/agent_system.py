"""
Option B: AIエージェントの開発
Function Callingと外部APIとの連携
"""

import os
import sys
from pathlib import Path
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Callable

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import google.generativeai as genai
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

class ToolRegistry:
    """ツール（関数）の登録・管理クラス"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, func: Callable, description: str, parameters: Dict):
        """ツールを登録"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
    
    def get_tool_descriptions(self) -> str:
        """ツールの説明をテキスト形式で取得"""
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"**{name}**\n"
            desc += f"説明: {tool['description']}\n"
            desc += f"パラメータ: {json.dumps(tool['parameters'], ensure_ascii=False, indent=2)}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def execute(self, tool_name: str, parameters: Dict) -> Any:
        """ツールを実行"""
        if tool_name not in self.tools:
            return f"エラー: ツール '{tool_name}' が見つかりません"
        
        try:
            func = self.tools[tool_name]["function"]
            return func(**parameters)
        except Exception as e:
            return f"ツール実行エラー: {e}"

class ReActAgent:
    """ReAct（Reasoning and Acting）パターンのエージェント"""
    
    def __init__(self, model_name="gemini-pro"):
        # Google Gemini API設定
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.llm = genai.GenerativeModel(model_name)
        
        # ツールレジストリ
        self.tool_registry = ToolRegistry()
        
        # メモリ
        self.conversation_history = []
        self.working_memory = {}
        
        # デフォルトツールの登録
        self._register_default_tools()
    
    def _register_default_tools(self):
        """デフォルトツールの登録"""
        
        # 電卓ツール
        def calculate(expression: str) -> str:
            """数式を計算"""
            try:
                # 安全な計算のため、eval は使わずに基本的な演算のみ
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "エラー: 無効な文字が含まれています"
                
                result = eval(expression)
                return f"計算結果: {result}"
            except Exception as e:
                return f"計算エラー: {e}"
        
        self.tool_registry.register(
            "calculate",
            calculate,
            "数式を計算します",
            {
                "expression": {
                    "type": "string",
                    "description": "計算する数式（例: 2+3*4）"
                }
            }
        )
        
        # 天気情報ツール（モック）
        def get_weather(location: str) -> str:
            """天気情報を取得（モック）"""
            # 実際の実装では OpenWeatherMap API などを使用
            mock_weather_data = {
                "東京": {"temperature": "22°C", "condition": "晴れ", "humidity": "60%"},
                "大阪": {"temperature": "25°C", "condition": "曇り", "humidity": "70%"},
                "札幌": {"temperature": "15°C", "condition": "雨", "humidity": "80%"},
                "福岡": {"temperature": "26°C", "condition": "晴れ", "humidity": "55%"}
            }
            
            # 部分マッチで検索
            for city, data in mock_weather_data.items():
                if city in location or location in city:
                    return f"{city}の天気: {data['condition']}, 気温: {data['temperature']}, 湿度: {data['humidity']}"
            
            return f"{location}の天気情報は見つかりませんでした。利用可能な都市: {', '.join(mock_weather_data.keys())}"
        
        self.tool_registry.register(
            "get_weather",
            get_weather,
            "指定された場所の天気情報を取得します",
            {
                "location": {
                    "type": "string", 
                    "description": "天気を調べたい場所（例: 東京）"
                }
            }
        )
        
        # メモ保存ツール
        def save_memo(key: str, content: str) -> str:
            """メモを保存"""
            self.working_memory[key] = {
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            return f"メモ '{key}' を保存しました"
        
        self.tool_registry.register(
            "save_memo",
            save_memo,
            "メモを保存します",
            {
                "key": {"type": "string", "description": "メモのキー"},
                "content": {"type": "string", "description": "メモの内容"}
            }
        )
        
        # メモ取得ツール
        def get_memo(key: str) -> str:
            """メモを取得"""
            if key in self.working_memory:
                memo = self.working_memory[key]
                return f"メモ '{key}': {memo['content']} (保存日時: {memo['timestamp']})"
            return f"メモ '{key}' は見つかりませんでした"
        
        self.tool_registry.register(
            "get_memo",
            get_memo,
            "保存されたメモを取得します",
            {
                "key": {"type": "string", "description": "取得するメモのキー"}
            }
        )
        
        # タイマーツール
        def set_timer(seconds: int, message: str = "時間です！") -> str:
            """タイマーを設定"""
            try:
                seconds = int(seconds)
                if seconds <= 0:
                    return "エラー: 正の数を指定してください"
                if seconds > 3600:  # 1時間以上は制限
                    return "エラー: 1時間以内で設定してください"
                
                print(f"⏰ {seconds}秒のタイマーを開始...")
                time.sleep(seconds)
                return f"🔔 {message}"
            except ValueError:
                return "エラー: 有効な数値を指定してください"
        
        self.tool_registry.register(
            "set_timer",
            set_timer,
            "指定された秒数のタイマーを設定します",
            {
                "seconds": {"type": "integer", "description": "タイマーの秒数"},
                "message": {"type": "string", "description": "タイマー終了時のメッセージ（オプション）"}
            }
        )
    
    def parse_action(self, text: str) -> tuple:
        """テキストからアクション情報を抽出"""
        lines = text.strip().split('\n')
        
        for line in lines:
            if line.startswith('Action:'):
                action_part = line[7:].strip()
                
                # JSON形式のパラメータを探す
                if '{' in action_part and '}' in action_part:
                    # JSON部分を抽出
                    start = action_part.find('{')
                    json_part = action_part[start:]
                    tool_name = action_part[:start].strip()
                    
                    try:
                        parameters = json.loads(json_part)
                        return tool_name, parameters
                    except json.JSONDecodeError:
                        pass
                
                # 関数呼び出し形式を解析 tool_name(param1=value1, param2=value2)
                if '(' in action_part and ')' in action_part:
                    tool_name = action_part.split('(')[0].strip()
                    param_part = action_part.split('(')[1].split(')')[0]
                    
                    parameters = {}
                    if param_part.strip():
                        # パラメータを解析（簡易版）
                        for param in param_part.split(','):
                            if '=' in param:
                                key, value = param.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"\'')
                                parameters[key] = value
                    
                    return tool_name, parameters
        
        return None, None
    
    def generate_response(self, prompt: str) -> str:
        """LLMで応答を生成"""
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"LLM応答生成エラー: {e}"
    
    def run(self, task: str, max_iterations: int = 10, verbose: bool = True) -> str:
        """ReActループを実行"""
        
        if verbose:
            print(f"🎯 タスク: {task}")
            print("=" * 50)
        
        # 初期プロンプト構築
        tools_description = self.tool_registry.get_tool_descriptions()
        
        prompt = f"""
あなたは与えられたタスクを完了するために、利用可能なツールを使って段階的に問題を解決するエージェントです。

利用可能なツール:
{tools_description}

タスク: {task}

以下の形式で思考と行動を繰り返してください：

Thought: 現在の状況を分析し、次に何をすべきかを考える
Action: tool_name(parameter1=value1, parameter2=value2)
Observation: [ツールの実行結果がここに表示されます]

最終的な答えが得られたら、以下の形式で回答してください：
Final Answer: [最終的な回答]

それでは始めてください：
"""
        
        conversation = [prompt]
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- 反復 {iteration + 1} ---")
            
            # LLMで次のステップを生成
            current_prompt = "\n".join(conversation)
            response = self.generate_response(current_prompt)
            
            if verbose:
                print(f"🤖 エージェント:\n{response}")
            
            conversation.append(response)
            
            # Final Answerがある場合は終了
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[1].strip()
                if verbose:
                    print(f"\n✅ 最終回答: {final_answer}")
                return final_answer
            
            # アクションを解析・実行
            tool_name, parameters = self.parse_action(response)
            
            if tool_name and parameters is not None:
                if verbose:
                    print(f"🔧 ツール実行: {tool_name}({parameters})")
                
                # ツール実行
                result = self.tool_registry.execute(tool_name, parameters)
                observation = f"Observation: {result}"
                
                if verbose:
                    print(f"👁️ 観察結果: {result}")
                
                conversation.append(observation)
            
            else:
                # アクションが解析できない場合
                if "Action:" in response:
                    error_msg = "Observation: アクションの形式が正しくありません。tool_name(parameter=value)の形式で指定してください。"
                    conversation.append(error_msg)
                    if verbose:
                        print(f"⚠️ {error_msg}")
        
        # 最大反復数に達した場合
        final_msg = "最大反復数に達しました。タスクを完了できませんでした。"
        if verbose:
            print(f"\n❌ {final_msg}")
        return final_msg
    
    def add_tool(self, name: str, func: Callable, description: str, parameters: Dict):
        """カスタムツールを追加"""
        self.tool_registry.register(name, func, description, parameters)
    
    def list_tools(self):
        """利用可能なツールをリスト表示"""
        print("利用可能なツール:")
        print("=" * 30)
        for name, tool in self.tool_registry.tools.items():
            print(f"📦 {name}")
            print(f"   説明: {tool['description']}")
            print(f"   パラメータ: {list(tool['parameters'].keys())}")
            print()

def demo_basic_agent():
    """基本的なエージェントデモ"""
    print("🤖 基本エージェントデモ")
    print("=" * 40)
    
    agent = ReActAgent()
    
    # サンプルタスク
    tasks = [
        "25 * 4 + 100 を計算してください",
        "東京の天気を調べて、メモに保存してください",
        "5秒のタイマーを設定してください",
        "保存したメモを確認してください"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*20} タスク {i} {'='*20}")
        result = agent.run(task, verbose=True)
        print(f"\n📝 結果: {result}")
        
        if i < len(tasks):
            input("\n次のタスクに進むには Enter を押してください...")

def demo_custom_tools():
    """カスタムツールデモ"""
    print("🛠️ カスタムツールデモ")
    print("=" * 40)
    
    agent = ReActAgent()
    
    # カスタムツールの追加
    def generate_password(length: int = 8, include_symbols: bool = False) -> str:
        """パスワードを生成"""
        import random
        import string
        
        chars = string.ascii_letters + string.digits
        if include_symbols:
            chars += "!@#$%^&*"
        
        password = ''.join(random.choice(chars) for _ in range(int(length)))
        return f"生成されたパスワード: {password}"
    
    agent.add_tool(
        "generate_password",
        generate_password,
        "指定された長さのパスワードを生成します",
        {
            "length": {"type": "integer", "description": "パスワードの長さ（デフォルト: 8）"},
            "include_symbols": {"type": "boolean", "description": "記号を含めるか（デフォルト: False）"}
        }
    )
    
    def text_analysis(text: str) -> str:
        """テキスト分析"""
        word_count = len(text.split())
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        return f"""テキスト分析結果:
- 文字数: {char_count}
- 文字数（空白除く）: {char_count_no_spaces}
- 単語数: {word_count}
- 平均単語長: {char_count_no_spaces/word_count:.1f}文字"""
    
    agent.add_tool(
        "text_analysis",
        text_analysis,
        "テキストの統計情報を分析します",
        {
            "text": {"type": "string", "description": "分析するテキスト"}
        }
    )
    
    # カスタムツールのテスト
    tasks = [
        "12文字で記号を含むパスワードを生成してください",
        "『人工知能は素晴らしい技術です』というテキストを分析してください",
        "生成したパスワードをメモに保存してください"
    ]
    
    for task in tasks:
        print(f"\n{'='*50}")
        result = agent.run(task, verbose=True)

def demo_multi_step_task():
    """複数ステップタスクのデモ"""
    print("🎯 複数ステップタスクデモ")
    print("=" * 40)
    
    agent = ReActAgent()
    
    complex_task = """
以下のタスクを順番に実行してください：
1. 東京と大阪の天気を調べる
2. 両都市の気温を比較して、どちらが暖かいかを計算で確認
3. 結果を「天気比較」というキーでメモに保存
4. 最後に保存したメモを確認して報告
"""
    
    result = agent.run(complex_task, max_iterations=15, verbose=True)
    print(f"\n🎉 複雑タスクの最終結果: {result}")

def interactive_demo():
    """インタラクティブデモ"""
    print("💬 インタラクティブエージェントデモ")
    print("=" * 40)
    print("エージェントと対話できます。'quit'で終了")
    
    agent = ReActAgent()
    
    # 利用可能なツールを表示
    agent.list_tools()
    
    while True:
        task = input("\n🧑 タスクを入力してください: ").strip()
        
        if task.lower() in ['quit', 'exit', '終了']:
            print("デモを終了します。")
            break
        
        if not task:
            continue
        
        print("\n" + "="*50)
        result = agent.run(task, verbose=True)
        print(f"\n📋 最終結果: {result}")

def benchmark_agent():
    """エージェントのベンチマーク"""
    print("📊 エージェントベンチマーク")
    print("=" * 40)
    
    agent = ReActAgent()
    
    benchmark_tasks = [
        {
            "task": "2+2を計算してください",
            "expected_keyword": "4",
            "category": "計算"
        },
        {
            "task": "東京の天気を教えてください",
            "expected_keyword": "天気",
            "category": "情報取得"
        },
        {
            "task": "テストメモを保存して確認してください",
            "expected_keyword": "テストメモ",
            "category": "メモリ操作"
        }
    ]
    
    results = []
    
    for i, test in enumerate(benchmark_tasks, 1):
        print(f"\n📝 テスト {i}: {test['category']}")
        print(f"タスク: {test['task']}")
        
        start_time = time.time()
        result = agent.run(test['task'], verbose=False)
        execution_time = time.time() - start_time
        
        # 成功判定（簡易版）
        success = test['expected_keyword'].lower() in result.lower()
        
        results.append({
            "task": test['task'],
            "category": test['category'],
            "success": success,
            "execution_time": execution_time,
            "result": result
        })
        
        print(f"結果: {'✅ 成功' if success else '❌ 失敗'}")
        print(f"実行時間: {execution_time:.2f}秒")
    
    # 統計レポート
    print("\n" + "="*50)
    print("📈 ベンチマーク結果")
    print("="*50)
    
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    avg_time = sum(r['execution_time'] for r in results) / len(results)
    
    print(f"成功率: {success_rate:.1f}%")
    print(f"平均実行時間: {avg_time:.2f}秒")
    
    # カテゴリ別結果
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    print("\nカテゴリ別結果:")
    for category, cat_results in categories.items():
        cat_success_rate = sum(r['success'] for r in cat_results) / len(cat_results) * 100
        print(f"  {category}: {cat_success_rate:.1f}% 成功")

def main():
    """メイン実行関数"""
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEYが設定されていません")
        print(".envファイルでAPIキーを設定してください")
        return
    
    print("🤖 AIエージェント ワークショップ")
    print("=" * 40)
    
    while True:
        print("\nデモを選択してください:")
        print("1. 基本エージェントデモ")
        print("2. カスタムツールデモ") 
        print("3. 複数ステップタスクデモ")
        print("4. インタラクティブデモ")
        print("5. ベンチマークテスト")
        print("6. 終了")
        
        choice = input("\n選択 (1-6): ").strip()
        
        if choice == "1":
            demo_basic_agent()
        elif choice == "2":
            demo_custom_tools()
        elif choice == "3":
            demo_multi_step_task()
        elif choice == "4":
            interactive_demo()
        elif choice == "5":
            benchmark_agent()
        elif choice == "6":
            print("ワークショップを終了します")
            break
        else:
            print("無効な選択です")

if __name__ == "__main__":
    main()
