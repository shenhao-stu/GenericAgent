> **实现状态 — Round 3（2026-06-01）：全部 12 项已实现并经 Monitor 对抗验证。**
> 重构先行（高内聚/低耦合，符合 code_review_principles + qianxuesen 系统设计）：5 个 god-file 拆分到位（main.rs 2505→607、app/mod.rs 2078→705、components/mod.rs→cockpit/、overlay.rs→overlay/、render.rs<600），`AppEvent` 总线切断 UI↔bridge 传输耦合（`send_active` 归零）；**331 测试全绿 / 0 警告 / 4 条 parity 不变量守住**。
>
> 逐项落地（√ = 代码+渲染证据确认；**live** = 需你实跑确认观感）：
> - **Q1** √ 历史输入整条 rgb(58,58,58) + `❯` 前缀 · **Q2** √ 默认关鼠标捕获→终端原生拖选复制、去 ctrl+shift+c、复制走 `block.source` 无 `\n` · **Q3** √ 行内/块级 latex 原子换行不再打碎 + 流式 holdback + 边到边实时渲染 · **Q4** √ spinner ↑↓ token 实时
> - **Q5** √ 左键进 session/右键回对话 · **Q6** √ 对齐 v2 快捷键 + shift/ctrl+enter 换行 + 命令去重（别名标记） · **Q7** √ `❯❯ GenericAgent` slogan + llm·model·dir·session 头 + 输出框上方完成行 + 下方两行(session info / ⎿ Tips) + 去 `❯ chat` + `/keybindings` · **Q8** √ 工具展开/折叠 summary + 点击 ▸ 折叠任意节点 + 零 "Turn N"
> - **Q9** √ spinner 只用 `⠿`、脸只在 pet、tab 默认 bear · **Q10** √ 命令不重复/中英文齐/`/scheduler` 真反射模式(9 reflect + 8 cron，无假 09:00)/`/mouse` 默认开并移除/`/continue` 重放全部历史去 ✅ · **Q11** √ 主题改名 `default` + 深浅色重设计 + 每命令边框/字符特效(goal=Pulse◆ / hive=Orbit⬡ / conductor=Sweep▸ / morphling=Rainbow◆) · **Q12** √ 彩蛋/tips 双语 + `@` 缓存提速 + ctrl+s 不卡
>
> **live 待你实跑确认（headless 无法判定的观感）：** 输入 `/goal /hive /conductor /morphling` 时的边框+字符特效配色与流动是否优雅；你终端里拖选复制是否顺手；整体审美是否到位。**不满意的项告诉我，继续迭代直到满足。**

## 原始需求

1. 所有历史的输入应该是整条进行颜色渲染[58,58,58]，然后应该包含❯ [例如❯ /copy]，目前的版本并没有❯，请你增加一下。
2. 目前tui_v4.py的依旧无法进行终端的选中文本复制[然后不应该有ctrl+shift+c作为终端所有复制的功能，这完完全全的不合理]。同时我希望支持后能否进行无换行的复制，参考tuiapp_v2.py的复制逻辑，能够正常复制markdown的表格，以及多行复制，不会出现/n的情况。
3. 目前markdown和latex公式依旧不能进行正常的渲染，而且我希望支持实时渲染，而不是整体输出完之后的渲染。
4. 目前运行态◠ 思考中… (3.2秒 · ↑1.2k ↓340 · ctx ▰▰▱▱▱ 48%)，中的token输入和输出↑↓没有进行数字的实时动态渲染和懒加载变动。
5. 目前tui_v4的布局和快捷键存在非常严重的问题，理论上应该是输入框为空的时候左键进入session会话面板，现在绑定成了右键。请你进行修复。
6. 请你仔细对齐所有tuiapp_v2.py的快捷键的绑定逻辑，我希望tui_v4完全包含tuiapp_v2所绑定的快捷键。注意换行目前请你绑定shift+enter或者ctrl+enter, 然后是目前一些/命令可以进行去重。启用D:\GenericAgent\memory\code_review_principles.md删除重复冗余的代码和命令。
7. 目前整体的页面初始化可以参考tui_v4以及claude code的风格进行设计D:\Screenshots\clip_20260601_011003.png，请你将>_ GenericAgent换成更具有设计感且简洁的slogan。然后展示llm的渠道[简洁版本]，model, diectory, session[session_name]。然后输出框上方，当模型输出完成后显示，当前session总运行时间和token消耗，参考tuiapp_v2的设计[⠿ Patched for 1m 46s  · ↑ 47.3k · ↓ 472]。然后输入框下方，显示两行，一行为当前运行态session的信息，llm，model，effort，ctx, branch,第二行为⎿ Tips,能进行动态渲染和更新。不要出现❯ chat等丑陋的内容了。左侧对齐。快捷键可以单独绑定一个/keybindings或者ctrl+/进行弹出。
8. 然后模型整体的输出过程请你参考tuiapp_v2以及tui_v4,以及D:\GenericAgent\temp\claude-code和https://github.com/openai/codex源码。应该是展开为工具调用，然后折叠为summary内容，注意你可以点击展开summary小箭头▸的任何内容查看详细的过程[保证渲染的稳定性]。注意注意不要出现Turn 1 ...等的非常丑陋的渲染了。
9. 然后关于emoji只出现pet，不要出现spinner，spinner就只用⠿就好了。同时终端tab的显示。默认用bear。
10. 请你确保所有的/命令不会出现各种重复的情况，同时所有的功能都是可用的，然后保证中英文切换时正常的，/scheduler应该包含所有的反射模式。目前似乎缺少了很多内容。请你也保证所有命令的ui渲染都时正常的，例如/status, /workflows, /rewind。目前/mouse似乎不需要的，应该是默认开启的。同时/continue后，没有正常加载所有历史的对话，只有» ✅ 已恢复 148 轮完整对话（model_responses_335438.txt），而且我之前非常明确说过不要用✅这种icon。
11. 关于/theme请你进行深入的调研，重新设计几个适合黑底和白色的theme主题，同时命名进行更改，不要用ga-default,就用default就行了。关于/effects，请你重新进行设计，目前只有彩虹色的边框方案，请你寻找更多的方案。适配到不同的命令中，/goal /hive /conductor /morphling。同时当输入这些命令的时候，这些字符"/goal"本身也会进行特效的渲染，请你设计几个各自不同有区分度，简洁优雅，适合各自特点的边框和字符特效。
12. 注意tips和Sleuthing…等等的彩蛋请你继续补充，也注意中英文的版本，然后注意@的渲染尽可能快速，ctrl+s不会出现卡死的情况。