# 参考文献与竞品分析

## 直接竞品

### UGCS — Uncertainty-Guided Checkpoint Selection
- Nguyen et al., 2025, arXiv:2511.09864
- 链接: https://arxiv.org/abs/2511.09864
- **核心机制**: ANLL (uncertainty) 找 hardest samples → 平均它们的 training reward
- **差异**: UGCS 是 RL fine-tuning 特有（依赖 per-sample reward），PAAS 通用（SFT/RL/merging）
- **理论**: 无 bound
- **模型**: ≤1B（Qwen2.5-0.5B, Falcon3-1B, Qwen3-0.6B）
- **PAAS 对比**: 机制不同（uncertainty ranking vs alignment monitoring），UGCS 需作为 baseline

### EST — Evaluator Stress Test (Proxy Gaming Detection)
- Shihab et al., 2025, arXiv:2507.05619; ACL 2026 Findings
- 链接: https://arxiv.org/abs/2507.05619
- **核心机制**: 扰动不变性测试检测 proxy gaming
- **差异**: EST 是 detection + corrective intervention（format penalty/judge randomization/data filtering），不做 checkpoint selection
- **验证结果**: 确认不构成 selection 竞争

### WARM — Weight Averaged Reward Models
- Ramé et al., 2024, arXiv:2401.12187 (143 引用)
- **核心机制**: Weight-space averaging of reward models 以缓解 reward hacking
- **差异**: 解决 reward model 本身的设计，不是 checkpoint 选择
- **验证结果**: 不构成 selection 竞争

## 相邻工作

### Checkpoint Merging via Bayesian Optimization
- Liu et al., 2024, arXiv:2403.19390 (31 引用)
- Merging checkpoint 用的 BO，不是 proxy-quality 分析

### Robust Fine-Tuning of Zero-Shot Models
- Wortsman et al., 2022, CVPR (1185 引用)
- Model soup 奠基工作，但未讨论 proxy-quality misalignment

## 关键 Gap 定位

现有工作无人做「在线监控 proxy alignment → 触发谨慎选择 → bound-aware 聚合」这个完整组合。

## 领域活跃子方向

目前最接近工作集中在 RL fine-tuning 语境，PAAS 覆盖的 SFT + model merging 场景更广泛但相关竞争也更少。
