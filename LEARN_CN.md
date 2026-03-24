# MiniMind 完整代码学习指南

> 目标：从零理解一个小型 LLM 是如何从零开始被训练出来的。本文档面向对大语言模型有初步了解、但不了解完整训练流程的读者。

---

## 一、项目总览

MiniMind 是一个**从零训练小型大语言模型**的完整复现项目。它包含了训练一个大模型必经的所有阶段：

| 阶段 | 核心目标 | 训练脚本 |
|------|----------|----------|
| **Pretrain（预训练）** | 学知识：让模型学会"词语接龙" | `train_pretrain.py` |
| **SFT（监督微调）** | 学对话：教会模型按照指令格式回答 | `train_full_sft.py` |
| **LoRA（轻量微调）** | 低成本微调：用少量参数适配垂域 | `train_lora.py` |
| **Distillation（蒸馏）** | 让小模型学习大模型的知识 | `train_distillation.py` |
| **DPO（直接偏好优化）** | 用人类偏好数据对齐模型 | `train_dpo.py` |
| **PPO/GRPO/SPO（RLAIF）** | 强化学习对齐：AI 给模型打分来优化 | `train_ppo.py` / `train_grpo.py` / `train_spo.py` |

> **为什么叫"词语接龙"？** 预训练的本质是：给模型一段文本，让它预测下一个词。比如输入"中国的首都是北"，模型输出"京"，这就是"接龙"。

---

## 二、目录结构

```
minimind/
├── model/                      # 模型定义
│   ├── model_minimind.py       # MiniMind 核心模型架构（重点！）
│   ├── model_lora.py           # LoRA 低秩适配实现
│   └── tokenizer.json/config   # 分词器配置
├── dataset/                    # 数据集处理
│   └── lm_dataset.py          # 四种数据集类的实现（重点！）
├── trainer/                    # 训练脚本
│   ├── trainer_utils.py        # 训练工具函数（断点续训、初始化等）
│   ├── train_pretrain.py      # 预训练
│   ├── train_full_sft.py      # 全参数 SFT
│   ├── train_lora.py          # LoRA 微调
│   ├── train_distillation.py  # 知识蒸馏
│   ├── train_dpo.py           # DPO 偏好优化
│   ├── train_ppo.py           # PPO 强化学习
│   ├── train_grpo.py          # GRPO 强化学习
│   └── train_spo.py           # SPO 强化学习
├── scripts/                    # 工具脚本
│   ├── convert_model.py       # PyTorch ↔ Transformers 格式互转
│   └── serve_openai_api.py   # 兼容 OpenAI API 的服务接口
├── eval_llm.py               # 模型推理与对话（最重要！用它测试效果）
├── requirements.txt / pyproject.toml  # 依赖管理
└── README.md                 # 官方说明文档
```

---

## 三、模型架构详解（model_minimind.py）

这是整个项目最核心的文件。如果只能读一个文件，就读这个。

### 3.1 配置类：MiniMindConfig

```python
class MiniMindConfig(PretrainedConfig):
    def __init__(self,
        hidden_size: int = 512,        # 词嵌入维度（类比：每个词的"特征向量"多长）
        num_hidden_layers: int = 8,     # Transformer 层数（类比：模型"深度"）
        num_attention_heads: int = 8,   # 注意力头数（类比：并行思考8个方面）
        num_key_value_heads: int = 2,   # KV 头数（GQA：省显存的关键）
        vocab_size: int = 6400,         # 词表大小（6400个词，基本够用）
        rope_theta: int = 1000000,      # RoPE 旋转位置编码基数
        use_moe: bool = False,          # 是否用 MoE 专家混合架构
        ...
    )
```

**参数选择参考（ Scaling Law 的工程经验）**：
- `hidden_size=512, num_layers=8` → 约 **26M 参数**（MiniMind2-Small）
- `hidden_size=768, num_layers=16` → 约 **104M 参数**（MiniMind2）
- `hidden_size=640, num_layers=8 + MoE` → 约 **145M 参数**（MiniMind2-MoE）

> **为什么不用更大的模型？** 因为目标是在单卡 3090（24GB）上训练。参数量大到一定程度显存就放不下了。

### 3.2 核心模块

#### RMSNorm（均方根归一化）

```python
class RMSNorm(torch.nn.Module):
    def _norm(self, x):
        # 核心：除以 RMS（均方根），而不是均值
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

类比：把一组数字"标准化"，让它们的整体规模一致。比 LayerNorm 少计算均值，节省一点算力。

#### RoPE（旋转位置编码）

```python
def precompute_freqs_cis(dim, end, rope_base=1e6):
    # 对每个维度配一个"旋转角度"
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # 外积：每个位置 × 每个频率
    # 最终返回 cos 和 sin，在 attention 时给 Q、K 旋一个角度
```

**为什么需要 RoPE？** 模型需要知道"词在第几位"。绝对位置编码（如 `pos=5`）难以泛化到训练没见过过的长度。RoPE 通过让每个位置的向量旋转一个角度，使**相对位置关系**自然编码在向量中——两个词无论相隔多远，它们的旋转角度差值始终反映它们的距离。

#### Attention（多头注意力）

```python
class Attention(nn.Module):
    def forward(self, x, position_embeddings, past_key_value=None, ...):
        xq = self.q_proj(x)   # Query：我在找什么
        xk = self.k_proj(x)   # Key：我有什么特征
        xv = self.v_proj(x)   # Value：我的实际内容

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 加上位置信息

        # 如果有之前的 KV cache，只拼接新的
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 核心：Q @ K^T / sqrt(d) → softmax → @ V
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        output = F.softmax(scores, dim=-1) @ xv
```

**KV Cache 的作用**：生成文本时，已计算过的 Key 和 Value 不必重新算，直接从 cache 里取即可。节省约一半的计算量。

#### MoE（混合专家）门控

```python
class MoEGate(nn.Module):
    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)  # 打分
        scores = logits.softmax(dim=-1)                     # softmax 转概率
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k)  # 选 top-k 专家
        return topk_idx, topk_weight, aux_loss  # 哪个专家来处理，权重多大
```

**MoE 的核心思想**：不是每个 token 都让全部参数计算，而是通过门控选择部分专家处理。比如 4 个专家，每次只选 2 个激活——这样 1 次前向只用了 2/4 的 FFN 参数，但理论上模型可以学到更多知识。

### 3.3 完整模型：MiniMindForCausalLM

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        self.model = MiniMindModel(config)  # 主体
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 技巧：词嵌入层和输出层共享权重（ tying weights），省一半参数量

    def forward(self, input_ids, labels=None, ...):
        hidden_states, past_key_values, aux_loss = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            # 计算 CE Loss，但注意我们预测的是"下一个 token"
            shift_logits = logits[..., :-1, :]    # 扔掉最后一个
            shift_labels = labels[..., 1:]         # 扔掉第一个（错位对齐）
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

> **为什么叫 Causal（因果）？** 因为模型在预测第 `t` 个 token 时，只能看到 `1~t` 的历史，不能看到未来的 token。这是通过 attention mask（causal mask）实现的。

---

## 四、数据处理详解（dataset/lm_dataset.py）

数据是 LLM 训练的燃料。数据格式不对，训练就会失败。

### 4.1 预训练数据 PretrainDataset

```python
class PretrainDataset(Dataset):
    def __getitem__(self, index):
        text = self.samples[index]['text']  # {"text": "白日依山尽..."}
        tokens = tokenizer(text).input_ids          # 转成数字 [323, 1042, ...]
        tokens = [bos_id] + tokens + [eos_id]      # 加上开始/结束标记
        # padding 到固定长度
        input_ids = tokens + [pad_id] * (max_len - len(tokens))
        labels = input_ids.clone()                 # 标签和输入一样
        labels[pad位置] = -100                      # padding 不算入 loss
        return input_ids, labels
```

**为什么要 padding 到固定长度？** GPU 喜欢整齐的批量数据，就像排队要站整齐才能一起处理。

### 4.2 SFT 数据 SFTDataset

预训练是"学知识"，SFT 是"学对话格式"。数据长这样：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]
}
```

关键处理：`generate_labels()` 函数会**只对 assistant 的回答部分计算 loss**，user 部分不计算——这叫"教师forcing"（Teacher Forcing），只让模型学习"给定问题后怎么回答"，不学习"怎么提问"。

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)  # 先全部标记为忽略
    # 找到所有 <bos>assistant 开头到 <eos> 结尾的位置
    # 只在这段区间内 labels = input_ids（计算 loss）
    for j in range(start, end):
        labels[j] = input_ids[j]
    return labels
```

### 4.3 DPO 数据 DPODataset

DPO 需要**成对偏好数据**——同一个问题，有好的回答（chosen）和差的回答（rejected）：

```json
{
  "chosen": [{"role":"user","content":"Q"}, {"role":"assistant","content":"good answer"}],
  "rejected": [{"role":"user","content":"Q"}, {"role":"assistant","content":"bad answer"}]
}
```

### 4.4 RLAIF 数据 RLAIFDataset

RLAIF 不需要人类标注的偏好，而是让模型自己生成多个回答，再让奖励模型（Reward Model）打分。

```json
{"conversations": [{"role":"user","content":"问题"}, {"role":"assistant","content":"（无，实际推理时由模型生成）"}]}
```

assistant 的内容在训练时由**当前策略模型实时采样生成**，这就是 On-Policy 强化学习的核心特征。

---

## 五、训练流程详解

### 5.1 预训练（train_pretrain.py）—— 学知识

**目标**：让模型学会"给定前文，预测下一个词"。

核心训练循环：

```python
for step, (input_ids, labels) in enumerate(loader):
    with autocast(dtype=bfloat16):        # 混合精度，加速+省显存
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss     # aux_loss 是 MoE 的负载均衡loss

    scaler.scale(loss).backward()          # 反向传播
    scaler.step(optimizer)                 # 参数更新
    scaler.update()
```

**Loss 组成**：
- `res.loss`：标准语言模型交叉熵——模型预测下一个词的概率分布，与真实词的距离
- `res.aux_loss`：MoE 专用——防止某些专家被过度使用（"负载均衡"）

**关键训练参数**：
- `batch_size=32`：每批处理 32 个序列
- `accumulation_steps=8`：实际 batch = 32×8 = 256（梯度累积，等效大 batch）
- `max_seq_len=340`：每个序列最大 340 个 token（中文字符约 500~600 个）
- `learning_rate=5e-4`：预训练学习率较高
- `grad_clip=1.0`：梯度裁剪，防止梯度爆炸

### 5.2 SFT（train_full_sft.py）—— 学对话

SFT 和预训练结构几乎一样，区别在于：
1. **数据不同**：SFT 用对话数据，预训练用纯文本
2. **Loss mask 不同**：SFT 只在 assistant 回复部分算 loss
3. **学习率更低**：通常 `1e-4` 到 `5e-5`

### 5.3 LoRA（train_lora.py）—— 低成本微调

**核心思想**：不更新全部参数，只更新少量低秩矩阵。

```python
# LoRA 的本质：在原始的 W (d×k) 旁边，并行加两个小矩阵 A (d×r) 和 B (r×k)
# 更新方式：W' = W + A @ B（训练时冻结 W，只更新 A 和 B）
```

LoRA 的参数效率极高——假设 rank=8，全模型 100M 参数，LoRA 可能只占用 0.1M。

```python
apply_lora(model)  # 注入 LoRA 权重
# 冻结非 LoRA 参数
for name, param in model.named_parameters():
    param.requires_grad = ('lora' in name)  # 只有 lora 开头的参数可训练
```

### 5.4 知识蒸馏（train_distillation.py）—— 师徒传承

**核心思想**：不仅让小模型学习"标准答案"，还让它学习大模型的"思维方式"（softmax 概率分布）。

```python
# 损失函数 = α × CE_loss + (1-α) × KL_divergence
teacher_probs = F.softmax(teacher_logits / T, dim=-1)   # 教师概率分布
student_log_probs = F.log_softmax(student_logits / T, dim=-1)
distill_loss = F.kl_div(student_log_probs, teacher_probs) * T^2
```

`T`（温度）越大，概率分布越平滑，能学到更多"软知识"。

### 5.5 DPO（train_dpo.py）—— 偏好学习

**核心思想**：不需要单独的 Reward Model，直接用偏好对学习。

```python
# DPO 损失：直接最大化"chosen 优于 rejected"的概率
ref_logps = ref_model(input_ids).log_probs
pi_logps = model(input_ids).log_probs

# 策略项：actor - ref
loss = -log_sigma(β * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))
```

> **DPO vs PPO**：DPO 是离线学习（数据提前准备好），PPO 是在线学习（边生成边训练）。DPO 更稳定，PPO 更灵活。

### 5.6 GRPO（train_grpo.py）—— 群体相对策略优化

**核心思想**：对同一个问题，模型生成 N 个回答，打分后高于均值的被鼓励，低于均值的被抑制。无需单独训练 Critic 网络。

```python
rewards = reward_model.get_score(responses)       # N 个回答各自打分
grouped = rewards.view(-1, num_generations)     # 分组
mean_r, std_r = grouped.mean(), grouped.std()   # 组内均值和标准差
advantages = (rewards - mean_r) / (std_r + 1e-4)  # 相对优势
```

---

## 六、评估与推理（eval_llm.py）

训练完了怎么用？`eval_llm.py` 提供了两种方式：

```bash
# 1. 自动测试（内置 prompt）
python eval_llm.py --load_from ./MiniMind2

# 2. 手动输入对话
python eval_llm.py --load_from ./MiniMind2
# 输入 1 进入手动模式，然后可以自由对话
```

关键参数：
- `--weight`：指定用哪个阶段的模型（`pretrain`/`full_sft`/`dpo`/`reason`）
- `--lora_weight`：是否外挂 LoRA 权重
- `--max_new_tokens`：最大生成长度
- `--temperature`：控制随机性（0=确定性的，1=高度随机）
- `--historys`：携带几轮历史对话

---

## 七、项目配置对照表

| 模型 | hidden_size | num_layers | 参数量 | 适用场景 |
|------|-------------|------------|--------|----------|
| MiniMind2-Small | 512 | 8 | 26M | 快速实验 |
| MiniMind2-MoE | 640 | 8 + MoE | 145M | 兼顾效果与体积 |
| MiniMind2 | 768 | 16 | 104M | 最佳效果 |

---

## 八、训练推荐方案

### 最快复现 Zero 模型（约 2 小时 + 3 元）

```bash
# 预训练
cd trainer
python train_pretrain.py

# SFT
python train_full_sft.py

# 测试
cd ..
python eval_llm.py --weight full_sft
```

### 完整复现 MiniMind2（需要多卡）

```bash
# 预训练（2 epochs）
torchrun --nproc_per_node 8 train_pretrain.py

# SFT（2 epochs）
torchrun --nproc_per_node 8 train_full_sft.py

# DPO
torchrun --nproc_per_node 8 train_dpo.py

# RLAIF
torchrun --nproc_per_node 8 train_grpo.py
```

---

## 九、核心代码流程图

```
用户输入文本
     │
     ▼
Tokenizer（分词）───> 数字 ID 序列
     │
     ▼
MiniMindForCausalLM.forward()
     │
     ├── MiniMindModel（主体 Transformer）
     │       │
     │       ├── Embedding（词嵌入 + RoPE 位置编码）
     │       │
     │       ├── ×N 个 MiniMindBlock
     │       │       │
     │       │       ├── RMSNorm（预处理归一化）
     │       │       ├── Attention（多头注意力 + KV Cache）
     │       │       ├── RMSNorm（后处理归一化）
     │       │       └── FeedForward（或 MoE FFN）
     │       │
     │       └── RMSNorm → 隐藏状态
     │
     └── lm_head（线性层：隐藏状态 → 词表概率分布）
     │
     ▼
Softmax → 下一个词的概率
     │
     ▼
Sampling（采样）───> 生成下一个 token
     │
     ▼
Tokenizer（解码）───> 人类可读文本
```

---

## 十、常见问题

**Q: 为什么词表只有 6400？**
A: 词表越小，词嵌入层参数越少，整体模型越小。中文可以用字符级或子词级分词，6400 个 token 对中文来说压缩率虽然不如 qwen/llama，但足够覆盖常用表达。

**Q: 为什么用 GQA（Grouped Query Attention）？**
A: 标准 MHA（Multi-Head Attention）每个 head 都有独立的 K、V。GQA 让多个 Q head 共享一组 K、V 头——例如 8 个 Q head、2 个 KV head，K/V 计算量减少到 1/4。显存节省显著，尤其对长序列生成至关重要。

**Q: 什么是 `logits_to_keep`？**
A: 推理时不需要对整个序列重新计算 logits，只需保留最后一步的输出。`logits_to_keep` 参数控制只计算最后几个 token 的 logits，大幅加速推理。

**Q: 断点续训是怎么实现的？**
A: `lm_checkpoint()` 函数自动保存模型权重、优化器状态、学习率调度器状态、epoch 和 step 到 checkpoint 文件。重启时检测到文件就自动加载恢复。
