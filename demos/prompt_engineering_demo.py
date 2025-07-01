"""
LLMの基礎デモ：プロンプトエンジニアリング
異なるプロンプトでの出力の違いを体験
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# 環境変数読み込み
load_dotenv()

# Google Gemini APIの設定
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def demo_prompt_engineering():
    """プロンプトエンジニアリングのデモ"""
    
    # 基本的なタスク
    task = "以下の文章を要約してください：\n人工知能（AI）は、コンピューターシステムが人間のような知的行動を示す技術分野です。機械学習、自然言語処理、コンピュータビジョンなどの技術を組み合わせて、複雑な問題を解決します。近年、深層学習の発展により、AIの性能は飛躍的に向上しています。"
    
    # 異なるプロンプトパターン
    prompts = {
        "基本": task,
        
        "具体的指示": f"""
{task}

要件：
- 50文字以内で要約
- 専門用語は平易な言葉に置き換え
- 箇条書きは使わない
""",
        
        "役割設定": f"""
あなたは小学生にもわかりやすく説明するのが得意な先生です。
{task}
小学生でも理解できるように、簡単な言葉で要約してください。
""",
        
        "例示付き": f"""
{task}

要約の例：
良い例：「AIは人間のように考えるコンピューター技術で、最近とても進歩している」
悪い例：「人工知能という技術がある」

上記の良い例を参考に要約してください。
""",
        
        "思考過程": f"""
{task}

まず、以下の手順で考えてください：
1. 重要なキーワードを特定
2. 主要なポイントを整理
3. 簡潔にまとめる

思考過程：
"""
    }
    
    print("=" * 60)
    print("プロンプトエンジニアリング デモ")
    print("=" * 60)
    
    for prompt_type, prompt in prompts.items():
        print(f"\n【{prompt_type}】")
        print("-" * 40)
        print(f"プロンプト:\n{prompt}")
        print("-" * 40)
        
        try:
            response = model.generate_content(prompt)
            print(f"出力:\n{response.text}")
        except Exception as e:
            print(f"エラー: {e}")
        
        print("=" * 60)

def compare_temperature_settings():
    """Temperature設定による出力の違いを比較"""
    
    prompt = "AIの未来について100文字程度で予測してください。"
    temperatures = [0.0, 0.5, 1.0]
    
    print("\n【Temperature設定による違い】")
    print("=" * 60)
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 30)
        
        # Geminiの場合、generation_configを使用
        generation_config = genai.types.GenerationConfig(
            temperature=temp,
            max_output_tokens=200
        )
        
        for i in range(3):  # 3回生成して違いを確認
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                print(f"試行{i+1}: {response.text}")
            except Exception as e:
                print(f"エラー: {e}")
        
        print("=" * 60)

def few_shot_learning_demo():
    """Few-shot Learningのデモ"""
    
    print("\n【Few-shot Learning デモ】")
    print("=" * 60)
    
    # Zero-shot
    zero_shot_prompt = """
以下の文の感情を「ポジティブ」「ネガティブ」「ニュートラル」で分類してください。

文：今日のプレゼンテーションは完璧でした！
"""
    
    # Few-shot
    few_shot_prompt = """
以下の例を参考に、文の感情を「ポジティブ」「ネガティブ」「ニュートラル」で分類してください。

例1：
文：素晴らしい映画でした！
分類：ポジティブ

例2：
文：最悪の一日だった...
分類：ネガティブ

例3：
文：天気は曇りです。
分類：ニュートラル

文：今日のプレゼンテーションは完璧でした！
分類：
"""
    
    prompts = {
        "Zero-shot": zero_shot_prompt,
        "Few-shot": few_shot_prompt
    }
    
    for prompt_type, prompt in prompts.items():
        print(f"\n【{prompt_type}】")
        print(f"プロンプト:\n{prompt}")
        print("-" * 40)
        
        try:
            response = model.generate_content(prompt)
            print(f"出力: {response.text}")
        except Exception as e:
            print(f"エラー: {e}")
        
        print("=" * 60)

def chain_of_thought_demo():
    """Chain of Thought推論のデモ"""
    
    print("\n【Chain of Thought デモ】")
    print("=" * 60)
    
    problem = "太郎は5個のりんごを持っています。花子に2個あげて、次郎から3個もらいました。太郎は今何個のりんごを持っているでしょうか？"
    
    # 通常の回答
    normal_prompt = f"{problem}"
    
    # Chain of Thought
    cot_prompt = f"""
{problem}

ステップバイステップで考えてみましょう：
"""
    
    prompts = {
        "通常": normal_prompt,
        "Chain of Thought": cot_prompt
    }
    
    for prompt_type, prompt in prompts.items():
        print(f"\n【{prompt_type}】")
        print(f"プロンプト:\n{prompt}")
        print("-" * 40)
        
        try:
            response = model.generate_content(prompt)
            print(f"出力:\n{response.text}")
        except Exception as e:
            print(f"エラー: {e}")
        
        print("=" * 60)

def main():
    """メイン実行関数"""
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEYが設定されていません。")
        print(".envファイルを確認してください。")
        return
    
    print("LLM基礎デモ：プロンプトエンジニアリング")
    print("様々なプロンプト手法の効果を体験しましょう")
    
    # 各デモを実行
    demo_prompt_engineering()
    compare_temperature_settings()
    few_shot_learning_demo()
    chain_of_thought_demo()
    
    print("\nデモ完了！")
    print("プロンプトの書き方によって出力が大きく変わることが確認できました。")

if __name__ == "__main__":
    main()
