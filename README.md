# MiniMind-in-Depth 🌌
## 📖 项目简介
本项目是基于开源项目 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的实战复现与学习笔记。

这是一个用于 从零训练小型语言模型（MiniMind） 的本地仓库，支持：
* 预训练（Pretrain）
* 监督微调（Full SFT / LoRA）
* 偏好优化（DPO）
* 强化学习（PPO / GRPO）
* Streamlit Web Demo 推理

## 🛠️ 快速开始

### 1. 环境准备
```
# 安装依赖
pip install -r requirements.txt

# 确认 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"
```
### 2. 数据集准备
将训练数据[ Download ](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)放入 dataset/ 目录
```
dataset/
├── pretrain_hq.jsonl      # 预训练数据
├── sft_mini_512.jsonl     # SFT 数据
├── dpo.jsonl              # DPO 偏好数据
├── r1_mix_1024.jsonl      # RL / 推理相关数据
├── rlaif-mini.jsonl       # RLAIF / RL 数据
├── lm_dataset.py          # 数据加载逻辑
```

### 3. 目录结构说明
```
model/        # 模型结构定义
trainer/      # 各类训练脚本（pretrain / sft / dpo / rl）
dataset/      # 数据与数据加载逻辑
scripts/      # Web Demo / API / 工具脚本
eval_llm.py   # 简单推理测试
```

### 4. 训练流程
> Pretrain → SFT →（可选）DPO / PPO / GRPO
#### 4.1 预训练（Pretrain）
使用无标注文本训练基础语言模型。
```
python trainer/train_pretrain.py
```
使用的数据：`dataset/pretrain_hq.jsonl`

#### 4.2 监督微调（SFT）
让模型学会对话 / 指令跟随。
```
python trainer/train_full_sft.py
```
使用的数据：`dataset/sft_mini_512.jsonl`

#### 4.3 可选训练路线（Post-Training）
```
# LoRA 微调
python trainer/train_lora.py

# DPO（偏好优化）
python trainer/train_dpo.py

# PPO
python trainer/train_ppo.py

# GRPO
python trainer/train_grpo.py
```
使用的数据示例：
* `r1_mix_1024.jsonl`
* `rlaif-mini.jsonl`

### 5. 模型推理与测试
```
# 命令行测试
python eval_llm.py

# Web Demo
# 使用 Streamlit 启动本地对话界面：
cd scripts
streamlit run web_demo.py

# 浏览器访问：
http://localhost:8501
```























