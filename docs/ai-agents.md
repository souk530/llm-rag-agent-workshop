# AIエージェントの基礎

## 🤖 AIエージェントとは

AIエージェントは、環境を認識し、目標達成のために自律的に行動を選択・実行するAIシステムです。従来のLLMとは異なり、外部ツールを使って実際のタスクを実行できます。

## 🧭 エージェントの基本構成

### 1. コアコンポーネント
- **LLM**: 推論と意思決定の中枢
- **ツール**: 外部システムとのインターフェース
- **メモリ**: 過去の経験や文脈の保持
- **プランナー**: タスクの分解と実行順序の決定

### 2. 実行ループ
```
観察 → 思考 → 行動 → 観察 → ...
```

## 🛠️ Function Calling（関数呼び出し）

### 概念
LLMが適切なタイミングで外部関数を呼び出す仕組み

### 実装例
```python
import json
from typing import List, Dict

def get_weather(location: str) -> Dict:
    """指定された場所の天気情報を取得"""
    # 実際のAPI呼び出し
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "晴れ"
    }

def calculate(expression: str) -> float:
    """数式を計算"""
    try:
        return eval(expression)  # 実際の実装では安全な評価を使用
    except:
        return "計算エラー"

# 利用可能な関数の定義
available_functions = {
    "get_weather": {
        "function": get_weather,
        "description": "天気情報を取得します",
        "parameters": {
            "location": {"type": "string", "description": "場所"}
        }
    },
    "calculate": {
        "function": calculate,
        "description": "数式を計算します",
        "parameters": {
            "expression": {"type": "string", "description": "計算式"}
        }
    }
}
```

## 🔄 ReAct（Reasoning and Acting）パターン

### 概念
推論（Reasoning）と行動（Acting）を交互に実行するパターン

### フロー
1. **Thought**: 現在の状況を分析
2. **Action**: 必要な行動を選択・実行
3. **Observation**: 行動の結果を観察
4. 目標達成まで1-3を繰り返し

### 実装例
```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
    
    def run(self, task):
        max_iterations = 10
        
        for i in range(max_iterations):
            # Thought: 現在の状況を分析
            prompt = self._build_prompt(task)
            response = self.llm.generate(prompt)
            
            if "Action:" in response:
                # Action: ツールを実行
                action, input_data = self._parse_action(response)
                observation = self._execute_action(action, input_data)
                self.memory.append(f"Action: {action}({input_data})")
                self.memory.append(f"Observation: {observation}")
            
            elif "Final Answer:" in response:
                # タスク完了
                return self._extract_final_answer(response)
    
    def _execute_action(self, action_name, input_data):
        if action_name in self.tools:
            return self.tools[action_name]["function"](**input_data)
        return "Unknown action"
```

## 🌐 マルチエージェントシステム

### 概念
複数のエージェントが協調してタスクを実行するシステム

### アーキテクチャパターン

#### 1. 階層型（Hierarchical）
```
Manager Agent
├── Research Agent
├── Writing Agent
└── Review Agent
```

#### 2. パイプライン型（Pipeline）
```
Input → Agent A → Agent B → Agent C → Output
```

#### 3. 分散型（Distributed）
```
Agent A ←→ Agent B
    ↕        ↕
Agent C ←→ Agent D
```

### 実装例
```python
class AgentTeam:
    def __init__(self):
        self.agents = {
            "researcher": ResearchAgent(),
            "writer": WritingAgent(),
            "reviewer": ReviewAgent()
        }
    
    def collaborate(self, task):
        # 研究フェーズ
        research_data = self.agents["researcher"].research(task)
        
        # 執筆フェーズ
        draft = self.agents["writer"].write(research_data)
        
        # レビューフェーズ
        final_result = self.agents["reviewer"].review(draft)
        
        return final_result
```

## 🧠 メモリシステム

### 短期メモリ
- 現在のタスクセッション中の情報を保持
- 会話履歴、実行ログなど

### 長期メモリ
- セッション間で持続する知識
- ユーザーの好み、過去の成功例など

### 実装例
```python
class AgentMemory:
    def __init__(self):
        self.short_term = []  # 現在のセッション
        self.long_term = {}   # 永続化データ
        self.working_memory = {}  # 作業用
    
    def add_experience(self, action, result, success):
        experience = {
            "action": action,
            "result": result,
            "success": success,
            "timestamp": datetime.now()
        }
        self.short_term.append(experience)
        
        if success:
            # 成功例を長期メモリに保存
            self.long_term[action] = result
```

## 🔧 実用的なエージェント設計

### 1. タスクの分解
```python
def decompose_task(complex_task):
    """複雑なタスクを小さなステップに分解"""
    return [
        "情報収集",
        "分析・整理", 
        "案の作成",
        "検証・改善",
        "最終化"
    ]
```

### 2. エラーハンドリング
```python
def safe_execute(action, max_retries=3):
    for attempt in range(max_retries):
        try:
            return action()
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            time.sleep(1)  # リトライ前の待機
```

### 3. 進捗管理
```python
class TaskTracker:
    def __init__(self):
        self.completed_steps = []
        self.current_step = None
        self.remaining_steps = []
    
    def update_progress(self, completed_step):
        self.completed_steps.append(completed_step)
        if self.remaining_steps:
            self.current_step = self.remaining_steps.pop(0)
```

## 📊 パフォーマンス評価

### 成功率指標
- **タスク完了率**: 正常に完了したタスクの割合
- **精度**: 正しい結果を出した割合
- **効率性**: 平均実行時間、API呼び出し回数

### 評価方法
```python
def evaluate_agent(agent, test_cases):
    results = []
    for task in test_cases:
        start_time = time.time()
        result = agent.run(task)
        execution_time = time.time() - start_time
        
        results.append({
            "task": task,
            "result": result,
            "time": execution_time,
            "success": evaluate_success(task, result)
        })
    
    return calculate_metrics(results)
```

## 🚀 応用例

### 1. 自動化エージェント
- メール処理、スケジュール管理
- データ分析レポート作成
- コード生成・テスト

### 2. 知識労働支援
- 資料調査・まとめ
- 企画書作成支援
- 技術文書執筆

### 3. カスタマーサポート
- FAQ自動応答
- 問題解決支援
- エスカレーション判定

## 🔮 最新トレンド

- **Code Interpreter**: コード実行環境の統合
- **Vision Capabilities**: 画像・動画の理解と処理
- **Tool Learning**: 新しいツールの自動学習
- **Human-in-the-loop**: 人間との協調作業
