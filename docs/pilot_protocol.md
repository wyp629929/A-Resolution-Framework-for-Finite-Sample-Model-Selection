# Pilot 实验协议

## 目的

验证 PAAS 工程 pipeline 通不通，不是验证研究假设。

## 范围

1 seed × 1 schedule（cosine）× synthetic checkpoints（无 GPU 环境）

## 验收标准（P1-P6）

| ID | 标准 | 验证方式 |
|---|---|---|
| P1 | 训练产出 ≥2 checkpoint，step index 正确 | 检查 stages/checkpoints.json |
| P2 | Proxy 信号值域合理 | proxy_main ∈ [-0.5, 2.0] |
| P3 | Validation 信号值域合理 | val_main ∈ [0, 1] |
| P4 | ρ_t 落在 [-1, 1] | 所有 rho_spearman 在范围内 |
| P5a | 直接注入 trigger → cautious selection 路径正确 | selection_mode ≠ normal |
| P5b | 注入 mid-training trigger → fallback 到 safe checkpoint | last_safe_step 存在 |
| P6 | 全流程 < 30 分钟 | elapsed time |

## 已知限制（Pilot 阶段不解决）

1. **Synthetic fallback**: 无 GPU 时训练、信号抽取、评测全部走合成数据
2. **Bootstrap CI**: n=6 的 bootstrap 在小样本下不可靠，仅用于 pilot
3. **Self-judge bias**: ShareGPT 使用 LLaMA-3-8B self-judge，full-scale 必须切换为人工标注
4. **缺少真实 crossing**: 合成数据的 ρ_t 趋势是硬编码的，不是真实训练结果

## 标注停止规则（Full-scale）

- 均匀采样 6 个 checkpoint 先标注
- 用 permutation test 估计 ρ_t CI
- CI 宽度 < 0.15（在 Spearman ρ [-1,1] 尺度上）→ 停
- 否则加到 10 个点
- 不再增加

## 从 Pilot 到 Full-scale 的检查清单

- [ ] 真实 GPU 训练运行通过
- [ ] ρ_t 趋势可解释（非 NaN，非常数）
- [ ] 人工 pairwise 标注协议就绪
- [ ] CI 方法从 bootstrap 切换到 permutation
- [ ] AutoDL 资源配置确认
