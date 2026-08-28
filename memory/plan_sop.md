# Plan Mode SOP

> ⚠️ **【已弃用 / DEPRECATED】**
>
> 本 SOP 已弃用，不适合现阶段模型。**禁止再进入 plan mode、起探索 subagent、写 plan_XXX/plan.md。**
> 原 250 行协议（强制探索 subagent、checkbox 执行循环、强制 [VERIFY] 仪式）已删除，仅保留下方替代表与 §四 残桩。
>
> **维护动作**：若在 L1（`global_mem_insight.txt`）中发现对本 SOP（`plan_sop`）的任何索引，**立即删除该索引**，勿再导航到此文件。
>
> **请改用（按场景）：**
> | 场景 | 替代 | 入口 |
> |------|------|------|
> | 复杂多阶段 / 多 agent 编排 | `ultraplan_sop` | 仅当用户明确说 `ultraplan` / `UltraPlan` / `ultraplan mode` 时启用 |
> | 跨会话长期项目认知与记忆 | `project_mode_sop` | 用户要求「进入项目模式」或指定项目名时 |
> | 交付物是否站得住 | `deliverable_audit_sop` | 不可逆操作 / 对外交付 / 跨文件大改动时，起一个独立 subagent 对抗性复核 |
> | 其他任务 | 直接执行 | 无需任何 plan/mode；长任务用 `update_working_checkpoint` 保存关键上下文即可 |

---

## §四 验证态（残桩，仅供 plan 模式拦截器满足）

历史遗留：`ga.py` 的完成声明拦截器在 plan 模式下会把你导向「plan_sop §四」。若该拦截触发，只需做一件事：

1. 按 `deliverable_audit_sop.md` 起**一个**独立 subagent 做对抗性验证（角色=证明交付物不能用；每项检查必须有真实工具调用证据）。
2. 读它的 `result.md`，取最后一行 `VERDICT: PASS / FAIL / PARTIAL`。
3. PASS → 收尾；FAIL → 只修失败项后重验（最多 2 轮）；无 VERDICT 或全是叙述无工具输出 → 按 FAIL 处理。

不要因为这段残桩重建 plan.md 流程。正常任务根本不该进入 plan 模式。
