# /review SOP

## 何时使用

用户输入 `/review`，或要求"审一下这段改动 / 启用监察者 / code review"时。
触发后主 agent 在**当前 session** 内对抗审阅指定范围，把报告直接 echo 到对话。

## 启动

```
/review                  # 默认审本次 uncommitted 改动（主 agent 自跑 git diff 取范围）
/review <自然语言范围>   # 例：/review 关注 review_cmd.py 的 prompt 注入
/review help             # 用法
```

`GA_LANG=en` 切英文 prompt。

## 边界

- 只读：禁止 file_write / file_patch 任何业务代码
- 不开 subagent、不写 `review.md`、不打 `[ROUND END]`
- 报告 echo 完即结束，不再调任何工具

## 文件

- 入口：`frontends/review_cmd.py`
- 执行 prompt：`memory/review_sop/review_inline_prompt.txt` / `.en.txt`
- 评审原则：`memory/code_review_principles.md`（每条 finding 必须能映射其中一条）

## 输出

Verdict 规则：任一 P0 → `FAIL`；无 P0 但 ≥ 1 P1 → `CONDITIONAL`；否则 `PASS`。
字段细节（severity / location / evidence / fix / principle / confidence_score）见 `review_inline_prompt.txt`。
