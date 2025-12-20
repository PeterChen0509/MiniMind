# MiniMind-in-Depth 🌌
## 📖 项目简介
本项目是基于开源项目 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的实战复现与学习笔记。
原项目致力于以极低成本（单卡GPU+2小时）从零训练一个 LLM。本项目在此基础上，深入代码底层，对大语言模型构建的每一个核心环节进行了源码级的拆解与分析。
## 🚀 核心模块 (Core Features)
1. **基础组件构建 (Fundamentals)**
* **Tokenizer 训练**：从零构建分词器，理解词表生成与映射过程。
* **RMSNorm**：深入解析 Pre-Norm 机制及其在训练稳定性中的作用。
* **位置编码**：
    * 原始 Transformer 的绝对位置编码。
    * 主流 RoPE (Rotary Positional Embedding) 的数学原理与代码实现解析。
2. **模型架构剖析 (Model Architecture)**
* **Attention 机制详解**：
  * Multi-head Attention (MHA)
  * Grouped-query Attention (GQA)
  * Multi-query Attention (MQA) 的区别与代码实现。
* **混合专家模型 (MoE)**：拆解 Mixture of Experts 的路由机制与专家网络构建。
* **LLM 整体搭建**：如何将上述组件拼装成一个完整的 Decoder-only 语言模型。
3. **训练与微调流水线 (Training Pipeline)**
* **预训练 (Pretrain)**：LLM 知识注入的全流程代码解析。
* **监督微调 (SFT)**：指令跟随能力的训练实现。
* **强化学习与对齐 (Alignment)**：
    * **DPO (Direct Preference Optimization)**：直接偏好优化的源码级解析，理解如何无需 Reward Model 进行对齐。
4. **进阶优化与压缩 (Optimization)**
    * **LoRA 微调**：从零实现 Low-Rank Adaptation，不依赖第三方库的底层逻辑。
    * **模型蒸馏**：
      * **白盒蒸馏**：Logits 层面的知识迁移。
      * **黑盒蒸馏**：基于大模型输出数据的指令微调。
  
## 🛠️ 快速开始
```
# 安装依赖
pip install -r requirements.txt

# 预训练
python train_pretrain.py

# 指令微调
python train_full_sft.py
```

