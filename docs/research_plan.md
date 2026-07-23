# PAAS: Robust Model Selection under Proxy-Quality Misalignment

## 研究计划

### 论文核心主张

在 LLM 微调 / model merging 场景下，用 cheap proxy 信号（validation loss、win-rate）选 checkpoint 是标准做法，但已知会失灵。本文提出 **PAAS**：在线监控 proxy 与独立 validation 信号的排序一致性（ρ_t），一致性下降时触发 cautious selection，用窗口聚合替代 naive argmin。

### MVP 实验范围 (Tier 1)

| 模块 | 内容 |
|---|---|
| 主实验 | SFT (LoRA)，2 seeds × 2 schedules（cosine + constant） |
| 模型规模 | LLaMA-3-8B（主线），小模型可选补充 |
| Proxy 信号 | validation loss + win-rate（自动化） |
| Validation 信号 | GSM8K/CodeAlpaca 自动判定 + ShareGPT 人工 pairwise |
| 人工标注 | ≥2 人，Krippendorff's α 报告，均匀采样 6-8 checkpoint |
| Baseline | proxy-best + uniform ensemble + UGCS-adapted |
| 算法 | ρ_t 监控 + 分层聚合（proxy 定窗口，uniform/validation 定 winner） |
| 理论 | Proposition 1（简化 block bound），~4 页，标注"非 tight，motivation only" |
| Evaluation | 4-benchmark oracle（MMLU/HumanEval/MT-Bench/domain）+ FP/FN + τ 敏感性 |

### 关键设计约束

1. **Spearman ρ ∈ [-1,1]**，不归一化不取绝对值
2. **Validation signal 两层**：GSM8K/CodeAlpaca 全自动 + ShareGPT 人工 pairwise（对 anchor）
3. **Aggregation 分层**：proxy 只划候选窗口（粗粒度），窗口内 uniform 或 validation-guided 定 winner
4. **UGCS 适配保留两阶段结构**：筛 hard subset → 子集聚合，只替换 ANLL → entropy
5. **评估集跨模型规模一致**（1B/3B/8B 用同一套题目）

### Pilot 验收标准

| ID | 标准 |
|---|---|
| P1 | 训练产出 ≥2 checkpoint，step index 正确 |
| P2 | Proxy 信号值域合理（proxy_main ∈ [-0.5, 2.0]） |
| P3 | Validation 信号值域合理（val_main ∈ [0, 1]） |
| P4 | ρ_t 落在 [-1, 1] 且趋势可解释 |
| P5a | 强制触发能进入 cautious selection 代码路径 |
| P5b | Mock 分歧能正确 fallback 到上一个 safe checkpoint |
| P6 | 全流程 < 30 分钟 |

### Tier 2（可选加分项）

- 小模型（1B-3B）自然轨迹统计：crossing 常见性
- 8B 诱导 crossing 场景（domain shift）：保证失败案例数量
- Model merging case study：同一算法第二个应用
- UGCS 完整 SFT 移植 baseline

### Tier 3（不做实验，仅文字讨论）

- RL/GRPO 实验（related work 讨论差异）
- Chaining / 完整 correlated concentration bound
- 与论文 1/3 的关系（按 positioning 策略处理）
