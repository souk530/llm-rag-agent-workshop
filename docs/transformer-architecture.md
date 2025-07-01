# Transformerアーキテクチャの基礎

## 🧠 概要

Transformerは現代のLLMの基盤となるニューラルネットワークアーキテクチャです。2017年に発表された"Attention Is All You Need"論文で初めて紹介されました。

## 🔍 主要な構成要素

### 1. Attention機構

Attentionは、入力シーケンスの異なる部分に「注意」を向ける仕組みです。

#### Self-Attention
- 文中の各単語が他のすべての単語との関係性を学習
- 文脈理解が大幅に向上

#### Multi-Head Attention
- 複数の異なる「注意」の観点を並行して学習
- より豊かな表現が可能

### 2. トークン化と埋め込み表現

```python
# トークン化の例
text = "今日は良い天気です"
tokens = ["今日", "は", "良い", "天気", "です"]
token_ids = [1234, 56, 789, 101, 234]
```

### 3. 位置エンコーディング

- トークンの順序情報を保持
- Sinusoidal関数やLearnable Embeddingを使用

## 🔄 Transformerの動作フロー

1. **入力処理**: テキスト → トークン → 埋め込みベクトル
2. **エンコーダー**: Self-Attentionで文脈を理解
3. **デコーダー**: 次の単語を予測
4. **出力**: 確率分布から最適な単語を選択

## 💡 なぜTransformerが革新的だったのか

- **並列処理**: RNNと違い、全てのトークンを同時に処理可能
- **長距離依存**: 離れた単語同士の関係も効率的に学習
- **スケーラビリティ**: パラメータ数を増やしやすい設計

## 🛠️ 実装例

```python
import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        Q = self.w_q(x)  # Query
        K = self.w_k(x)  # Key
        V = self.w_v(x)  # Value
        
        # Attention計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output
```

## 📚 参考資料

- [Attention Is All You Need (原論文)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Transformerの数学的解説](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
