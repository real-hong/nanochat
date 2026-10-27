# nanochat + VS Code Codex：19 个学习任务提示词（修订版）

> 与仓库根目录的 `AGENTS.md` 配套。本文是**可调整的课程路线**，不是持久行为规则。
> 更新日期：2026-10-27。以你本地仓库当前 Commit 为最终依据。

## 使用方法

1. 将本文保存到 nanochat 仓库的 `study/ROADMAP.md`（不想保存也可直接从此文件复制提示词）。
2. 每个步骤建议开启一个新的 Codex 对话，但同一步骤中多个专题应在同一对话中连续完成；超长时可以提前换对话。
3. 每次直接复制对应步骤下方的**完整文本**，无须重复发送全局导师指令。每段都内置了工作区规则、版本与上下文检查。
4. **一个步骤通常需要多轮互动**。首次发送指令后，Codex 应仅开展第一专题；你可以继续要求“讲下一小节”“针对这段代码举例”或“开始考核”。
5. `study/PROGRESS.md` 与笔记**可选**，不存在时不妨碍授课；只有明确要求 Codex 保存时才修改文件。
6. 默认为源码只读分析。你明确要求进行某个小型实验后，才可以按照 `AGENTS.md` 的权限界限编写和运行实验；训练、下载、安装等资源操作另行审慎处理。

### 通用操作提示词（可选）

**学习步骤结束时发送：**

```text
请总结本步骤的实际学习结果：已掌握知识、关键源码与数学公式、我实际完成并验证的实验、考核证据、仍未解决的问题，以及下一步的前置条件。不要把建议实验写成已完成实验。先在对话中展示总结；只有我明确授权保存时，才写入 study/PROGRESS.md 和笔记文件。
```

**新对话需要衔接时发送：**

```text
请检查当前工作区 AGENTS.md；如果 study/PROGRESS.md 或相关笔记存在，先读取并概述与当前主题有关的已学知识和未解决问题。然后依照我随后发送的当前步骤提示词进行教学。不要假设笔记中未记录的学习内容已掌握。
```

---

## 学习路线索引

| 步骤 | 阶段 | 主题 |
|---|---|---|
| 0 | 准备阶段 | 工作区、版本与实验条件检查 |
| 1 | 第 1 阶段：整体与基础 | 项目全貌与 LLM 生命周期 |
| 2 | 第 1 阶段：整体与基础 | PyTorch 定向诊断 |
| 3 | 第 2 阶段：数据基础 | Tokenizer、BPE 与对话序列化 |
| 4 | 第 2 阶段：数据基础 | 预训练数据、Packing 与目标构造 |
| 5 | 第 3 阶段：模型核心 | GPT 配置、Embedding 与模型骨架 |
| 6 | 第 3 阶段：模型核心 | 从零推导 Causal Multi-Head Attention |
| 7 | 第 3 阶段：模型核心 | RoPE、GQA、QK Norm 和滑动窗口 |
| 8 | 第 3 阶段：模型核心 | MLP、Normalization、Residual 与 Block |
| 9 | 第 3 阶段：模型核心 | GPT.forward、Logits 与语言建模 Loss |
| 10 | 第 4 阶段：模型训练 | Pretraining Loop 与梯度累积 |
| 11 | 第 4 阶段：模型训练 | SGD、AdamW、Muon 与参数分组 |
| 12 | 第 4 阶段：模型训练 | Checkpoint、验证损失与评测指标 |
| 13 | 第 5 阶段：推理 | Autoregressive Generation 与 Sampling |
| 14 | 第 5 阶段：推理 | KV Cache、Prefill、Decode 与 Engine |
| 15 | 第 6 阶段：对齐训练 | Chat Template、SFT 与 Loss Masking |
| 16 | 第 6 阶段：对齐训练 | RL：REINFORCE、GRPO 与实际 chat_rl |
| 17 | 第 7 阶段：工程与综合 | 性能：FlashAttention、精度与分布式 |
| 18 | 第 7 阶段：工程与综合 | 独立实现并验证 Mini LLM |

---

## 准备阶段

### 步骤 0：工作区、版本与实验条件检查

**优先定位的源码：** `README.md、pyproject.toml、runs/runcpu.sh、git 工作区`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 0：工作区、版本与实验条件检查。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请对 nanochat 学习工作区做一次只读的准备检查。
1. 确认工作区根目录与 Git HEAD，报告 git status --short，并提醒我保留已有修改；核对 AGENTS.md 是否存在及其适用范围。
2. 阅读 README.md、pyproject.toml、runs/runcpu.sh，列出 Python/PyTorch/uv 的预期版本与可用执行模式；用只读命令检查本机 Python、PyTorch、CUDA/MPS/CPU 条件（如 Python 环境可用）。
3. 判断哪些章节只需要阅读源码，哪些微型实验能在 CPU 运行，哪些正式脚本会下载数据或消耗大量算力。
4. 给出一份资源风险表：可直接只读检查、需要先安装依赖、需要数据下载、长时间训练。
5. 我已有 AGENTS.md，禁止重复生成新的全局导师规则；本步骤不安装依赖、不修改源码、不下载数据、不启动训练。
验收：我清楚正在研究哪个版本、项目入口和当前设备能够支持哪些实验。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

## 第 1 阶段：整体与基础

### 步骤 1：项目全貌与 LLM 生命周期

**优先定位的源码：** `README.md、runs/speedrun.sh、runs/runcpu.sh、scripts/、nanochat/、tasks/`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 1：项目全貌与 LLM 生命周期。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请从系统架构角度介绍当前 nanochat 的完整生命周期，不要深入 Attention 数学细节。
A. 通过 README 与运行脚本，画出“原始文本→Tokenizer→训练 Batch→GPT 预训练→Checkpoint→SFT→RL（按实际流程）→推理”的总体流程；区分一定发生的流程与可选流程。
B. 列出关键入口脚本、核心模块、任务/评测模块及它们的调用关系。重点追踪一次预训练和一次聊天请求分别从哪个入口出发。
C. 解释 Base Model、Chat Model、Tokenizer、Evaluation、Inference Engine 的职责；比较 nanochat 与只覆盖预训练的 nanoGPT。
D. 给出“模块→关键文件/符号→输入→输出→上游/下游”的简洁表格，并指出后续源码阅读顺序和依赖。
E. 设计 5 道全局架构题，先问第一道，不要直接给答案。
验收：我能独立画出系统流程并知道出现问题时应查看哪个模块。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 2：PyTorch 定向诊断

**优先定位的源码：** `nanochat/gpt.py、scripts/base_train.py、nanochat/engine.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 2：PyTorch 定向诊断。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请依据当前 nanochat 源码开展面向 LLM 的 PyTorch 诊断，避免重复我已掌握的入门知识。
诊断题范围：shape/stride/contiguous；view/reshape/transpose/permute；matmul/broadcast；Embedding/Linear；softmax/cross_entropy；autograd、detach、no_grad；Module、state_dict；optimizer；dtype/device。
1. 从上述真实代码中各找出典型的使用场景，并说明掌握它们对后续阅读有什么价值。
2. 设计 8–10 道由易到难的诊断题，特别强调 Tensor Shape、广播和梯度传播。一次只出一道，等我回答。
3. 对于错误或不确定的概念，再给一个小 Tensor 手算示例和最小 PyTorch 验证代码；已掌握的直接跳过。
4. 最终只列出我的知识缺口、补习顺序和是否具备阅读 GPT.forward 的基础。
验收：我能准确追踪 (B,T,C)→(B,H,T,D)→(B,H,T,T) 的张量变换。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

## 第 2 阶段：数据基础

### 步骤 3：Tokenizer、BPE 与对话序列化

**优先定位的源码：** `nanochat/tokenizer.py、scripts/tok_train.py、scripts/tok_eval.py、tests/test_tokenizer.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 3：Tokenizer、BPE 与对话序列化。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请按“原理→手算→当前实现→最小实验”的顺序教授 Tokenizer。
A. 从字符/词/子词、byte-level BPE、Vocabulary Size、UNK 和可逆解码开始，使用一段小语料手工演示合并规则。
B. 查阅 tokenizer.py 和训练/评测脚本：定位 tokenizer 训练、encode/decode、Special Tokens、BOS、角色/工具消息序列化、保存与加载。区分包装库的实际职责，不预设版本 API。
C. 按 Token ID 列出一轮多消息对话编码的示意，并解释哪些是特殊 Token，哪些是普通文本。
D. 给一个我能独立实现的极简 BPE 实验规格；提供 round-trip 和边界测试，先不给完整答案。
E. 最后检验 BPE 训练与编码算法是否被真正理解。
验收：我能手工执行几次 BPE Merge，并解释文本到 Token ID 的转换。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 4：预训练数据、Packing 与目标构造

**优先定位的源码：** `nanochat/dataset.py、nanochat/dataloader.py、scripts/base_train.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 4：预训练数据、Packing 与目标构造。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请讲解真实数据是怎样变为 GPT 的输入与 Target 的。
1. 用条件概率解释 Causal LM / Next-token Prediction；给出输入 x、标签 y 错位构造的数学定义，说明 BOS、EOS/文档边界的影响。
2. 检查 dataset.py 与 dataloader.py，按实际函数追踪数据格式、分片读取、Tokenization、BOS-aligned Best-Fit、裁剪、Buffer、训练/验证划分与恢复状态。
3. 选用 B=2、T=5 和三份长度不同的玩具文档，手工演示一次 Packing、x/y 构造；讲清楚示例和真实实现的简化差异。
4. 设计最小 CPU DataLoader 实验：输出 (B,T) 的 x/y，验证 Target 偏移、文档边界与无意信息泄漏。
5. 最后提出一些覆盖 Packing 取舍和 Tensor Shape 的考核题。
验收：我能独立实现可测试的 Next-token Batch 构造器。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

## 第 3 阶段：模型核心

### 步骤 5：GPT 配置、Embedding 与模型骨架

**优先定位的源码：** `nanochat/gpt.py：GPTConfig、GPT、Block`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 5：GPT 配置、Embedding 与模型骨架。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请先只讲 GPT 的宏观计算图，不急着展开 Attention 细节。
1. 逐项核对 GPTConfig 的字段和有效组合（vocab_size、sequence_len、n_layer、n_head、n_kv_head、n_embd、window_pattern 等），解释约束与派生维度。
2. 阅读 GPT.__init__ 及参数初始化，讲 Token Embedding、Transformer Blocks、输出投影、位置处理、权重共享与否、初始化与 meta device（如果本版本使用）。
3. 用一个小配置（如 B=2、T=4、C=32、4 Heads、2 Layers）逐层追踪 Shape，粗略推导参数量，并验证估算的遗漏项。
4. 给出 GPT 的文本模块图和“常规 GPT 理论 vs 当前 nanochat 实现”的差异清单。
5. 设计一个不需要训练的参数数量核对实验；先解释接口和预期，不自动执行。
验收：给定配置，我能推出主要模块的形状与参数量。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 6：从零推导 Causal Multi-Head Attention

**优先定位的源码：** `nanochat/gpt.py：CausalSelfAttention`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 6：从零推导 Causal Multi-Head Attention。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

这是核心学习主题，请分几个逻辑小节展开：一次只讲一个小节。
A. 从 Q/K/V 的投影和点积相似度开始，严格推导 scaled dot-product attention，并解释 1/sqrt(d_k)、softmax 轴、加性 causal mask 与梯度的作用。
B. 解释 Multi-Head Attention 的拆分与合并：以 B=2、T=4、H=4、D=8 手工推导 (B,T,C)→Q/K/V→scores→weights→output 的所有 shape。
C. 使用小数值矩阵至少手算一行 masked attention 的结果，并与 PyTorch 基本算子等价实现对照。
D. 阅读实际 CausalSelfAttention；先在 n_head == n_kv_head 的普通 MHA 情形建立映射，再标记本项目对 Q/K/V、注意力内核的特殊处理，留待下一步深入。
E. 提供一个我自己编写的基础 Causal MHA 的接口、单元测试和边界条件，不直接提供完整实现。
验收：我能不参考源码独立写出基础 Causal MHA，并解释每个矩阵乘法。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 7：RoPE、GQA、QK Norm 和滑动窗口

**优先定位的源码：** `nanochat/gpt.py、nanochat/flash_attention.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 7：RoPE、GQA、QK Norm 和滑动窗口。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

在我已经掌握基础 MHA 的前提下，按四个独立专题教现代注意力改造：
专题 A RoPE：从二维旋转推导绝对角度差带来的相对位置性质；检查 apply_rotary_emb 的实际旋转约定、cos/sin 的构造与广播形状；写一个小规模数值测试。
专题 B GQA：比较 MHA/MQA/GQA；依据实际 n_head、n_kv_head 和 head_dim 追踪 Q、K、V 的 Shape，推导 KV Cache 占用以及约束条件。
专题 C QK Norm 和 Value Embedding：定位源码并解释它们如何影响 Q/K/V 与训练稳定性；明确哪些结论只能从代码观察，哪些属于设计推断。
专题 D Sliding Window：推导 Full Attention 与局部窗口的可见范围和复杂度；核对 window_pattern 的解析与最后一层策略，检查 FA3/SDPA 是否支持以及回退行为。
每个专题给出核心公式、代码位置、最小实验构想和 2 道检查题；一次只开始专题 A。
验收：我能实现简化 RoPE/GQA 并计算缓存大小、解释窗口掩码。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 8：MLP、Normalization、Residual 与 Block

**优先定位的源码：** `nanochat/gpt.py：norm、Linear、MLP、Block、GPT.forward`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 8：MLP、Normalization、Residual 与 Block。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请逐步拆解单个 Transformer Block。
1. 对比标准 FFN、ReLU²、GELU 与 SwiGLU，推导 MLP 的前向计算和 Shape；核查 nanochat 实际采用的激活。
2. 解释 LayerNorm/RMSNorm 的数学公式、归一化维度、可学习参数差异和数值稳定性；分析项目中的 norm 及 Linear dtype 行为。
3. 解释 Residual Connection、Pre-Norm、Post-Norm 的计算图和反向传播意义，直接追踪 Block.forward 的两条主路径。
4. 核查当前实现中的 Value Embedding、Smear/x0 residual、Residual Scaling、Backout 等机制；如果本版本不存在某项，要明确说明并用真实机制替代。
5. 让我独立实现一个简单的 Pre-Norm Transformer Block，要求给出输入输出契约、测试和梯度检查。
验收：我能画出 Block 计算图并独立实现简化版本。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 9：GPT.forward、Logits 与语言建模 Loss

**优先定位的源码：** `nanochat/gpt.py：GPT.forward`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 9：GPT.forward、Logits 与语言建模 Loss。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请完整追踪一条输入 Token ID 从 GPT.forward 到 Logits/Loss 的计算路径。
1. 先核对 forward 的真实签名、返回类型、targets 存在或缺失时的行为和其他控制参数。
2. 以 B=2、T=4、C=32、V=128 给出逐层 Tensor Shape 表：Embedding→Block→Norm/Output Head→Logits→Loss。
3. 从条件概率与最大似然严格推导 Cross Entropy / NLL，说明序列位置平均、ignore_index、loss_reduction、logit softcap（若存在）。
4. 设计可在 CPU 上运行的微型 forward/loss 实验：手工检查单位置 log-prob 与 torch.nn.functional.cross_entropy 的一致性；说明未实测时的预期。
5. 用综合考核确认我能解释每一步对梯度的影响。
验收：我能独立实现小 GPT 的 forward 和 loss，并正确理解预测形状。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

## 第 4 阶段：模型训练

### 步骤 10：Pretraining Loop 与梯度累积

**优先定位的源码：** `scripts/base_train.py、nanochat/dataloader.py、nanochat/common.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 10：Pretraining Loop 与梯度累积。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请按真实执行顺序分析预训练脚本。
A. 从最大似然、mini-batch SGD、前向、反向和参数更新建立训练闭环，区分 epoch、iteration、optimizer step、micro-batch。
B. 找到设备/分布式初始化、模型创建、数据迭代、Optimizer、学习率计划、训练循环、日志、评测及保存的对应代码位置。
C. 严格解释 device_batch_size、max_seq_len、total_batch_size（核对单位）、world_size、grad_accum_steps 的关系；推导平均 Loss 与梯度累积等价性，以及不同条件下的不等价情形。
D. 解释 warmup/warmdown、梯度缩放、梯度裁剪（如存在）、数值精度相关分支。
E. 指导我基于合成 Token 数据自己实现 10–30 次小步更新，记录 loss、grad norm、参数变化与 lr，先给实验方案和验收条件。
验收：我能自己写训练循环并解释每次权重更新是如何得到的。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 11：SGD、AdamW、Muon 与参数分组

**优先定位的源码：** `nanochat/optim.py、nanochat/gpt.py：setup_optimizer`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 11：SGD、AdamW、Muon 与参数分组。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请按优化算法依赖关系逐层讲解。
专题 A：从 SGD 到 Momentum，严格写出优化目标和更新方程；区分普通梯度下降和参数衰减。
专题 B：推导 Adam 的一阶/二阶矩、偏差修正、epsilon 与 AdamW decoupled weight decay；用单参数数值例子完成两次更新，设计和 torch.optim.AdamW 的对照实验。
专题 C：解释 Muon 处理矩阵权重更新的动机、正交化思想、Newton–Schulz 迭代（具体实现以源码为准），分清理论算法与编译/分布式细节。
专题 D：分析 setup_optimizer 怎样给 Embedding、输出头、矩阵参数及其他参数分组，以及不同学习率、权重衰减策略。
最后提出一项审查任务：给定一组参数，判断它们落在哪一组，解释为什么。
一次只讲专题 A；每个专题结束后我确认再进入下一个。
验收：我能手算 AdamW 更新，独立读懂项目的优化器分组。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 12：Checkpoint、验证损失与评测指标

**优先定位的源码：** `nanochat/checkpoint_manager.py、nanochat/loss_eval.py、nanochat/core_eval.py、scripts/base_eval.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 12：Checkpoint、验证损失与评测指标。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请讲解模型训练如何验证、保存以及恢复。
1. 根据源码列出实际 Checkpoint 结构：模型权重、Optimizer 状态、模型配置、训练步数、数据恢复位置、随机数状态（存在与否均需核对）。
2. 解释仅恢复模型权重与恢复完整训练状态的差异，以及设备/dtype/版本兼容问题。
3. 推导 Cross Entropy、Perplexity 与 Bits Per Byte 的定义和单位；解释它们各自适合比较什么、不能直接比较什么。
4. 阅读 CORE Evaluation 与其他评测脚本，区分训练指标、验证指标和实际任务表现。
5. 设计最小实验：训练几步→保存→重载→核对同一输入预测，继续一步训练并检验是否确实正确恢复；不得假定测试通过。
6. 最后设计一组诊断题：loss 降低是否意味着模型一定更好？什么情况 Checkpoint 恢复后结果变化？
验收：我可以解释并实现一个可靠的简化 Checkpoint/Validation 流程。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

## 第 5 阶段：推理

### 步骤 13：Autoregressive Generation 与 Sampling

**优先定位的源码：** `nanochat/gpt.py：generate、nanochat/engine.py、scripts/chat_cli.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 13：Autoregressive Generation 与 Sampling。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请从概率模型与真实推理源码两方面解释文本生成。
1. 推导 p(x_1,...,x_T)=∏p(x_t|x_<t)，说明自回归生成和训练目标如何对应。
2. 分别说明 Greedy、Temperature、Top-k、Top-p 的数学/实现机制；先核查哪几种在当前仓库中被实现，未实现的方法只做对照教学。
3. 跟踪基本 generate 或相应实际入口：从输入 Token、forward 的 Logits、末位置选择、softmax/multinomial、拼接 Token 到终止条件。
4. 用人工设置的 Logits 手算温度和 Top-k 对概率的影响，注明固定种子只能保证相同条件下可重现。
5. 给我编写基础 generate 函数的接口和测试，比较 Greedy 与随机采样，不引入 KV Cache。
验收：我能写无缓存的生成循环并解释各采样参数。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 14：KV Cache、Prefill、Decode 与 Engine

**优先定位的源码：** `nanochat/engine.py、nanochat/gpt.py、tests/test_engine.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 14：KV Cache、Prefill、Decode 与 Engine。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请分阶段讲解推理加速，重点在真实缓存结构和 Tensor Shape。
1. 说明无缓存逐 Token 解码为什么重复计算，区分 Prefill 和 Decode，推导基本计算量差异及长上下文限制。
2. 阅读当前 KVCache/Engine 的真实数据结构，追踪存储的 K/V、层数、Batch、序列位置、n_kv_head、head_dim、dtype；推导缓存显存占用。
3. 对一个短 Prompt 手动演示 Prefill 后 Cache 的内容，再演示两个 Decode 步骤；解释 GQA/滑动窗口下缓存的实际行为。
4. 分析 Engine 的 Batch 生成、停止条件、采样、工具调用（如适用）与缓存复用，明确哪些功能属于模型、哪些属于引擎。
5. 设计带缓存与不带缓存的微型 Attention 对照实验，比较同配置下的数值一致性与运行代价；必要时说明精度误差来源。
验收：我可以独立实现基本 KV Cache 并准确计算内存。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

## 第 6 阶段：对齐训练

### 步骤 15：Chat Template、SFT 与 Loss Masking

**优先定位的源码：** `scripts/chat_sft.py、nanochat/tokenizer.py、tasks/common.py、tasks/smoltalk.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 15：Chat Template、SFT 与 Loss Masking。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请从 Base Model 如何变为 Chat Model 的角度深入讲解 SFT。
A. 解释 Pretraining vs SFT、Instruction Data、Chat Template、System/User/Assistant/工具消息边界，核查本仓库支持的 Special Tokens 和实际序列化方式。
B. 分析 SFT 训练数据来源、任务混合、Batch 构造及 targets/loss mask：哪些 Token 参与监督、哪些不参与，逐 Token 演示一条多轮对话。
C. 跟踪 chat_sft.py 如何加载预训练模型和 Optimizer、设置训练参数、计算 Loss、评测与保存。
D. 推导 SFT Cross Entropy 与预训练 NLL 的关系，指出误掩码、序列截断、特殊 Token 错误等常见问题。
E. 设计一个几条问答数据的微型 SFT 实验，先让我独立写 mask 构造和测试，不下载真实 SFT 数据。
验收：我能正确对聊天数据生成 loss mask 并解释它。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 16：RL：REINFORCE、GRPO 与实际 chat_rl

**优先定位的源码：** `scripts/chat_rl.py、tasks/gsm8k.py、nanochat/engine.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 16：RL：REINFORCE、GRPO 与实际 chat_rl。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请分四个专题教授当前 nanochat 的强化学习实现。
专题 A 基础：从 Policy、Action、Reward、Expected Return 推导 log-derivative trick 与 REINFORCE；解释 Baseline/Advantage 如何影响梯度方差。
专题 B LLM 对应：Token 采样如何形成 trajectory，序列 Reward 如何分配到生成 Token；解释 On-policy、PPO、GRPO 和 KL regularization 的直观目的。
专题 C 真实实现：源码已声明其“GRPO”更接近简化的 REINFORCE。请核对 Reward、分组样本、Advantage、Token-level normalization、Loss Mask、参数更新、评测的准确行为，严禁把标准 GRPO 公式直接当成项目实际实现。
专题 D 玩具实验：一个离散策略及几个固定 reward，先手算梯度，再用很小的 PyTorch 模型验证高奖励 Action 的概率变化。
每个专题结束出题，确认后再推进。
验收：我能从数学目标推导梯度，并解释项目实际与经典算法的差异。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

## 第 7 阶段：工程与综合

### 步骤 17：性能：FlashAttention、精度与分布式

**优先定位的源码：** `nanochat/flash_attention.py、nanochat/common.py、nanochat/optim.py、scripts/base_train.py、scripts/infer_bench.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 17：性能：FlashAttention、精度与分布式。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

请以“瓶颈→优化方案→源码依据→代价”的方式研究性能，不要一开始陷入底层内核细节。
A. Attention：区分理论上 O(T²) 的权重矩阵、计算/显存访问的瓶颈；介绍 FlashAttention 的 IO-aware 思想；核对当前 FA3/SDPA 回退、dtype 与设备限制。
B. 精度：逐一解释 FP32/BF16/FP16、显存/吞吐/数值稳定性，核对本项目 COMPUTE_DTYPE、手动 cast、GradScaler、torch.compile 的用法。
C. 分布式：从 Data Parallel、DDP、All-Reduce 讲到 Optimizer State Sharding 和 Reduce-Scatter/All-Gather，只要求首先理解单机多卡的真实架构。
D. 度量：解释 FLOPs、FLOPS、tokens/s、MFU、峰值显存、prefill/decoding 吞吐，指出实际记录和 benchmark 入口。
E. 根据我的设备设计两个微型基准：比较不同 dtype/attention 实现的正确性与时间，但不要擅自跑重负载测试。
验收：给定瓶颈，我能判断它属于计算、带宽、显存还是通信限制。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

### 步骤 18：独立实现并验证 Mini LLM

**优先定位的源码：** `独立实验目录；参考 nanochat/gpt.py、dataloader.py、engine.py、checkpoint_manager.py`

**直接发送给 Codex：**

```text
现在开始 nanochat 学习计划的步骤 18：独立实现并验证 Mini LLM。

请先遵守当前工作区的 AGENTS.md；如果存在 study/PROGRESS.md 或相关已学笔记，请只读取与本步骤相关的部分。先核对本地 Commit 和下述目标文件/符号是否存在，不要把上游其他版本的实现当作本地事实；若路径或 API 已变化，说明差异并采用当前源码中的对应入口。

这是最终工程考核。请担任需求设计者与代码 Reviewer，不要替我一次生成完整项目。
先给出一个 CPU 可运行的 Mini LLM 规格，默认建议：vocab_size=128、T=32、n_embd=64、n_layer=2、n_head=4、n_kv_head=4、batch_size=8；以设备实际情况为准。
把工作拆成任务：简单 Tokenizer、Batch、Embedding、Causal MHA、RMSNorm、MLP/Block、GPT.forward、Cross Entropy、训练循环、验证、Checkpoint、Generation、可选 KV Cache。
对每个任务分别给：输入/输出契约、Shape、不变量、异常边界、单元测试、完成标准。一次只给我当前任务，等我自己实现后再审查。
最终验收必须涵盖：单元测试、有限差分或梯度合理性检查（必要时）、小数据集过拟合、loss 下降、checkpoint 重载一致性、可复现的生成实验。
最后将我的实现与 nanochat 按模块逐项比较，说明省略了哪些算法/工程设计，以及这些省略对正确性和性能的影响。
当前只输出任务拆解和第一个任务的验收要求，不创建任何项目文件。
验收：我能独立实现、解释、测试并调试整个训练与生成闭环。

请先给出本步骤的子专题顺序与源码定位结果，然后只开始第一个子专题；除非我明确要求完整讲解，否则不要一次性输出所有专题。
```

---

## 课程维护说明

- 这个路线的 0–18 编号只存在于 `ROADMAP.md` 中；不应把固定编号写回 `AGENTS.md`。
- 任何阶段发现项目源码发生变化，先调整该阶段的提示词再继续，不要强行适配旧的文件名或接口。
- 学习笔记只能记录真实掌握程度和已执行结果，避免由 Codex 自我认证“已掌握”。
- 对步骤 6–11、14、16 可以增加时间，掌握程度优先于完成速度。
