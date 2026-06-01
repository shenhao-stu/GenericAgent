# C5 — Command dedup · /scheduler reflect modes · /continue load-all · no-icon · i18n · eggs · perf

Audit of tuiapp_v4 against Q10/Q12. Every issue cites `file:line` in tui_v4; competitor patterns cite v2/v3/continue_cmd/Codex with paths. Judged vs `memory/code_review_principles.md`.

Scope reminder: research+audit only. The Rust sketches below are the *intended* edits; nothing was written to source.

---

## Findings (file:line bugs, root cause not symptom)

### F1 — `/continue` restores history but NEVER replays it into the transcript (Q10, the headline bug)
**Symptom**: after `/continue` Enter the user sees only `» ✅ 已恢复 148 轮完整对话（model_responses_335438.txt）` and zero conversation bubbles.
**Root cause**: the v4 `/continue` Enter handler `frontends/tuiapp_v4/src/main.rs:1144-1158` sends `UiToCore::Command{name:"restore", args:path}` and pushes the `continue.restoring` notice — full stop. The bridge handler `frontends/tuiapp_v4/scripts/ga_bridge.py:719-750` (`handle_restore`) calls `continue_cmd.restore(self._agent, path)` which (`frontends/continue_cmd.py:375-388`) ONLY does `_replace_backend_history(agent, history)` and returns the `✅ 已恢复 N 轮完整对话` string. It writes `backend.history` (so the *model* remembers) but emits exactly ONE system message frame — it never replays the *visible* turns. There is no transcript replay anywhere on the v4 path.
**The correct path already exists and is used by v3**: `continue_cmd.extract_ui_messages(path)` (`continue_cmd.py:514-553`) parses the log into `[{role, content}, …]` UI bubbles. v3 loops it after restore (`frontends/tui_v3.py:4127-4137`): `continue_cmd.restore(...)` then `for mm in continue_cmd.extract_ui_messages(path): _commit_user / _commit_assistant`. v2 does the same via `continue_extract(path)` → `sess.messages.append(ChatMessage(...))` (`frontends/tuiapp_v2.py:3980-3981`). v4 dropped this step.
**Why it can't be fixed UI-side only**: in v4 the *bridge owns the transcript* (the UI renders `MessageBegin/Delta/End` frames — `protocol.rs:41-53`). There is no "inject N bubbles" frame. So `handle_restore` must emit one `MessageBegin/Delta/End` triple per `extract_ui_messages` bubble. See Fix design D1.

### F2 — `✅` icon is hard-baked into the restore message (Q10: "我明确说过不要用 ✅ 这种 icon")
**Root cause**: the literal `✅`/`⚠️`/`❌` live in `continue_cmd.restore` return strings (`continue_cmd.py:380,382,388,391,395`) and the bridge forwards `msg` verbatim (`ga_bridge.py:749`). The whole tui_v4 surface is otherwise icon-disciplined (i18n strings carry no ✅), but this one string bypasses i18n entirely — it is generated Python-side and streamed as a system message. The de-icon must therefore happen in the **bridge** (it must not edit GA-core `continue_cmd.py` — that's shared by v2/v3/st/tg/dc/qt). See D2.

### F3 — `/scheduler` ships 4 FAKE hardcoded tasks; the real reflect modes are absent (Q10: "应包含所有反射模式，目前似乎缺很多")
**Root cause**: `frontends/tuiapp_v4/src/components/scheduler.rs:223-230` `default_tasks()` returns four invented rows (`daily standup reflect`/`inbox triage`/`weekly review`/`memory consolidation`) with invented cadences (`09:00`/`hourly`/`Fri 17:00`/`03:00`). None correspond to anything GA runs. The scheduler overlay is opened with these (`main.rs` `open_ui_command` "scheduler" arm at `:1369` forwards `Command{name:"scheduler"}` but the actual ReflectTask list is the hardcoded `default_tasks`).
**The canonical set is two-pronged** (from `frontends/slash_cmds.py`, the module v2 actually uses for `/scheduler`):
  1. **reflect-mode scripts** — `list_reflect_tasks()` / `list_launchable_services()` (`slash_cmds.py:249-343`): every `reflect/*.py` (non-`_`-prefixed), each launched as `[python, agentmain.py, --reflect, reflect/<f>]`. On THIS repo that is **9 modes**: `agent_team_worker, autonomous, bbs, bbs_monitor, checklist_master, goal_mode, scheduler, tg_monitor, trader` (the `--reflect SCRIPT` loader is `agentmain.py:193,239-276`).
  2. **cron tasks** — `list_scheduler_tasks()` (`slash_cmds.py:502-524`): every `sche_tasks/*.json` → `{name, path, schedule, enabled}`. On THIS repo that is **8 tasks**: `crypto_morning_brief, crypto_risk_audit, crypto_smart_money_scan, getoken_relay_monitor, linuxdo_monitor, linuxdo_monitor_glm, proxy_pool_cn_health, tg_monitor`.
The cron *cadence vocabulary* (the actual "reflect cadences") is in `reflect/scheduler.py:32-49` `_parse_cooldown`: `once · daily · weekday · weekly · monthly · every_<N>h/<N>m/<N>d`. Running-state detection is `slash_cmds.find_running()` (`slash_cmds.py:~397`, matches by cmdline tail). See D3 for wiring.

### F4 — `/mouse` is a removable command; mouse capture defaults ON, fighting native selection (Q10: "/mouse 似乎不需要，应默认开启…默认让终端原生选择可用")
**Root cause(s)**:
- Default is **capture ON**: terminal init unconditionally `EnableMouseCapture` (`main.rs:2273`), and `AppState` starts `mouse_capture` truthy (the `set_mouse_capture(app.mouse_capture)` toggle in `main.rs:556-558` flips from on). With capture on, the terminal can't do native drag-select-to-copy — the exact complaint.
- `/mouse` is a registered App command (`registry.rs:83`) AND there is already a redundant keychord `Ctrl+Shift+M` doing the same toggle (`main.rs:553-559`). The user says the command isn't needed.
The user's ask is: **default capture OFF** (native selection works out of the box), keep wheel-scroll reachable via the keychord only, and drop `/mouse` from the palette. Note: turning capture off loses wheel-scroll events — but v4 already has full keyboard scroll (PgUp/PgDn/End), and Q10 explicitly prefers native selection. See D4.

### F5 — `@` completion does a FULL uncached filesystem walk up to 3× per frame (Q12: "@ 的渲染尽可能快速")
**Root cause**: `app.list_project_files()` (`app/mod.rs:1188-1190`) calls `crate::input::paths::list_project_files(root)` (`input/paths.rs:208-245`) which walks the ENTIRE repo tree + re-parses `.gitignore` on every call, **with zero caching**. It is invoked, while an `@query` is live, from THREE places per frame:
  - `components/mod.rs:706` (`dropdown_height`, layout pass),
  - `components/mod.rs:724` (`render_dropdown`, paint pass),
  - `main.rs:1525` (the keystroke handler that opens the picker).
So at the 100ms redraw tick with `@` open, v4 walks all of `D:/GenericAgent` (incl. the huge `temp/`, plus everything not in `.gitignore`) ~3× per frame ≈ 30 walks/sec. `MAX_INDEXED_FILES` bounds the *output* but the walk still descends every non-ignored dir. THAT is the @ lag. (Codex solves the same problem with an async, debounced `FileSearchManager` — see Competitor patterns; v4 bans async so the fix is a cached snapshot, D5.)

### F6 — `Ctrl+S` cannot freeze on the dashboard toggle itself, but the seed-with-`@` path inherits F5 (Q12: "ctrl+s 不会出现卡死")
**Investigated root cause**: `Ctrl+S` → `app.open_dashboard()` (`main.rs:578-580` → `app/mod.rs:1324-1336`) only reads in-memory `dashboard_rows()` (no I/O) — and the dashboard *close* path (`main.rs:893-896`) is likewise pure. Bridge spawn for a new session (`bridge/mod.rs:337-407`) is `Command::spawn()` (non-blocking fork; stdout/stderr drained on background threads) so describing a task + Enter does not block on process start either. **There is no synchronous hang intrinsic to Ctrl+S.** The realistic freeze is a *render-cost* freeze: any path that drives the dashboard/cockpit while an `@query` is in the composer pays F5's 3×/frame tree-walk, which on this repo is multi-hundred-ms per frame and READS as a freeze. Fixing F5 (cache the file index) removes the only plausible Ctrl+S-adjacent stall. Net: F6 is a *consequence* of F5 — no separate handler bug, but the spec must verify it after D5. (If a residual stall remains, the next suspect is `snapshot_active_into_map` cloning a large transcript on every dashboard open — `app/mod.rs:1325` — currently a `Vec<Line>` clone; flag for measurement, not a confirmed bug.)

### F7 — command de-dup: three alias pairs are intentional, but the help panel must not double-list them (Q10: "确保所有 / 命令不重复")
**State**: `registry.rs:46-87` has 40 entries. The intentional aliases are: `sessions`≈`status` (`:48-49`), `stop`/`abort` (`:57-58`), `verbose`/`tools`/`trace` (`:79-81`), `quit`/`exit` (`:85-86`). These resolve fine and are NOT bugs. The risk is the `/help` and `/`-palette listing them as if distinct, and `did_you_mean` flapping between equidistant aliases. No *duplicate name* exists (the `registry_resolves_all_commands` test at `:271` would fail). So "dedup" here = (a) mark aliases so `/help` shows them as "alias of X", (b) confirm `palette_matches` doesn't surface, say, both `tools` and `trace` as separate primary hits. See D7. Also verify the §4 union count assertion `:291` stays consistent if any command (e.g. `mouse`) is removed per F4.

### F8 — `/scheduler` reflect-mode strings + the de-iconified restore + new tips have i18n holes (Q10/Q12 zh/en parity)
Detail in "i18n gaps" below. Summary: F3's new reflect-mode panel needs ~12 new keys (zh+en); F1/F2's restore replay needs a key for the (icon-free) restore banner; F4's removal needs the `/mouse`-less help to still describe native selection; F5's cache has no user-facing string (good). All must land in BOTH `EN_PAIRS` and `ZH_PAIRS` or the `dictionaries_cover_the_same_keys` test (`i18n/mod.rs:843`) fails.

### F9 — gerund pool is English-ONLY; tips exist in zh but "Sleuthing…" et al. never localize (Q12: "tips 和 Sleuthing 等彩蛋继续补充，注意中英文版本")
**Root cause**: `flavor/mod.rs:152-189` `GERUNDS` is 34 English words with a comment "English-only by design", and `gerund(tick)` (`:205`) ignores `Lang`. So a zh user running a turn sees `Sleuthing…`, `Reticulating…` etc. — untranslated. Tips ARE bilingual (`i18n/mod.rs:201-236`) but the gerunds (the more visible egg) are not. Q12 explicitly wants the eggs in both languages. See "tips/egg additions".

---

## Competitor patterns (CC / Codex / v2 / v3, with file cites)

### `/continue` load-all-history
- **v3 (the model to copy)** `frontends/tui_v3.py:4126-4137`: `_do_restore(path)` = `continue_cmd.restore(ag, path)` THEN `for mm in continue_cmd.extract_ui_messages(path): … self._commit_user(c) / self._commit_assistant(c)`, bracketed by `┄┄ continue_loading ┄┄` / `┄┄ continue_ready ┄┄` dim banners. No ✅.
- **v2** `frontends/tuiapp_v2.py:3980-3981`: `for h in continue_extract(path): sess.messages.append(ChatMessage(role=h["role"], content=h["content"]))` — same `extract_ui_messages` source, appended as real bubbles; also resets plan state so a stale planbar doesn't leak (`:3976-3994`).
- **continue_cmd** `frontends/continue_cmd.py:514-553` `extract_ui_messages`: each user-initiated round → one user bubble + one assistant bubble; auto-continuation LLM calls concat into the same assistant bubble with `**LLM Running (Turn N) ...**` markers; tool calls/results rendered in the same string shape `agent_loop` yields live, so fold logic folds them identically. This is the canonical replay parser — reuse it, don't reinvent.

### `/scheduler` reflect modes
- **slash_cmds.py** (what v2's `/scheduler` calls) `frontends/slash_cmds.py:249-277` `list_reflect_tasks()`, `:303-343` `list_launchable_services()` (hub.pyw parity, `cmd=[python, agentmain.py, --reflect, reflect/<f>]`), `:502-524` `list_scheduler_tasks()` (sche_tasks/*.json → name/schedule/enabled), `:346-390` `start_service()` (detached, poll-and-confirm), `:~397` `find_running()` (cmdline-tail match, "~30ms per /scheduler open"). Header tip `tuiapp_v2.py:157,160`: "/scheduler 调出 reflect 任务多选启动器" / "再点一下已勾选的任务可以 stop —— 取消勾选 = 停止" — confirming the multi-pick-toggle = start/stop UX v4 already mirrors structurally (good), just over the WRONG data.
- **reflect/scheduler.py:32-49** `_parse_cooldown`: the cadence grammar `once/daily/weekday/weekly/monthly/every_Nh|Nm|Nd`. `:62-131` `check()` is the cron engine (reads `sche_tasks/*.json`, `enabled`, `schedule HH:MM`, `repeat`, `max_delay_hours`). This is the source of truth for "what cadence labels are legal".

### `@` fuzzy file picker speed
- **Codex `FileSearchManager`** (`codex-rs/tui/src/`, in `App.file_search`, recon `codex_tui_patterns.md:35,55,136`): file search is an **async actor** driven by `AppEvent::StartFileSearch{query}` → background fuzzy search → `AppEvent::FileSearchResult` posted back to the loop. The composer never walks the FS on the render/keystroke thread; results arrive as events and are debounced. peer survey `peer_tuis_survey.md:227`: "[other TUIs] run completers off the event loop so heavy completers don't block." This is the gold standard.
- **Constraint for v4**: `tokio`/async is explicitly BANNED (`prior_v4_summary.md:34`, README:109/153 — "main loop stays synchronous; crossbeam channels + dedicated threads, not async tasks"). So v4 can't copy the async actor verbatim. The synchronous-equivalent is a **memoized index** (walk once, cache, refresh on a coarse timer / explicit invalidation) — which is also what CC does conceptually: CC debounces the statusline hook 300ms and aborts in-flight (`claude_code_patterns.md:32`) rather than recomputing per keystroke.

### tips / status-word eggs
- **Codex** `tooltips.rs` + `tooltips.txt` (recon `codex_tui_patterns.md:119-128`): tips loaded once via `lazy_static`+`include_str!`, rotated; a remote announcement channel can ship tips without a release. Shimmer header is the visible "working" affordance (`shimmer.rs`, `:111`). Codex's status word is a shimmering "Working" (not a gerund pool) — v4's 34-gerund pool is actually richer/closer to CC.
- **CC** ships the gerund/"status word" pool (Pondering/Reticulating/Sleuthing…) — v4 copied this into `flavor/mod.rs:154-189` verbatim. CC's are English-only too; v4 SHOULD do better per Q12 (bilingual).

### command de-dup / registry shape
- **Codex `slash_command.rs`** (fossies mirror; recon notes): a single `enum SlashCommand` with `strum` derives (`AsRefStr`/`EnumIter`/`EnumString`), `description()` per variant, iterated via `SlashCommand::iter()` — one source of truth, no parallel description table to drift. v4's `COMMANDS: &[SlashCommand]` const array (`registry.rs:46`) is the moral equivalent and already single-source — good. v4's gap is only *alias presentation*, not duplication.
- **v2 `COMMANDS`** `tuiapp_v2.py:1211-1242`: the canonical 26-row union (`/help …/quit`). v4's 40 rows are that set + multi-session (`switch/close/branch/rename`) + flavor (`emoji/effort/effects/theme/language/verbose+aliases`) + `mouse`. The v2 list does NOT contain `/mouse`, `/verbose`, `/tools`, `/trace`, `/emoji`, `/effort`, `/effects`, `/theme`, `/language`, `/workflows`, `/scheduler`(v2 has it via slash_cmds), `/abort` — those are v3/v4 additions. So removing `/mouse` (F4) is consistent with the v2 baseline.

---

## Fix design (Rust sketches: the actual changed lines / new fn signatures)

### D1 — replay restored bubbles into the v4 transcript (Python, `ga_bridge.py` only — GA core untouched)
In `frontends/tuiapp_v4/scripts/ga_bridge.py` `handle_restore` (currently `:738-750`), after `continue_cmd.restore` succeeds, emit one message triple per UI bubble:
```python
with self._turn_lock:
    continue_cmd = _import_continue_cmd()
    msg, _is_full = continue_cmd.restore(self._agent, path)
    bubbles = []
    try:
        bubbles = continue_cmd.extract_ui_messages(path)   # [{role, content}, …]
    except Exception:
        _eprint("[ga_bridge] extract_ui_messages failed:\n" + traceback.format_exc())
# Replay the prior conversation into the visible transcript (v3 parity:
# tui_v3.py:4129). Each bubble = one MessageBegin/Delta/End triple.
for b in bubbles:
    content = _bound(str(b.get("content") or ""))
    if not content.strip():
        continue
    role = "user" if b.get("role") == "user" else "assistant"
    bmid = self._new_mid()
    self.emit({"type": "MessageBegin", "mid": bmid, "role": role})
    self.emit({"type": "MessageDelta", "mid": bmid, "text": content})
    self.emit({"type": "MessageEnd", "mid": bmid, "reason": "stop"})
# THEN the (de-iconified) restore banner as the final system line.
mid = self._new_mid()
self.emit({"type": "MessageBegin", "mid": mid, "role": "system"})
self.emit({"type": "MessageDelta", "mid": mid, "text": _deicon_restore(msg, len(bubbles))})
self.emit({"type": "MessageEnd", "mid": mid, "reason": "stop"})
```
UI side: the bubbles arrive as ordinary `MessageBegin/Delta/End` and render through the existing transcript path — **no Rust change required** for the replay itself. Optionally update the `continue.restoring` notice copy. Caveat: very long sessions emit many frames; bound to e.g. the last `N` bubbles if 148-turn replays are too heavy (the bridge already `_bound`s each text). The user explicitly wants ALL turns, so default to all and only cap on a measured problem.

### D2 — strip the ✅/⚠️/❌ icon in the bridge (do NOT edit continue_cmd.py)
Add a tiny helper in `ga_bridge.py` and route the banner through it (used in D1):
```python
def _deicon_restore(msg, n):
    # continue_cmd returns "✅ 已恢复 N 轮完整对话（name）\n(…)"; the user wants no ✅.
    # Strip a leading status glyph + surrounding space; keep the informative text.
    s = (msg or "").lstrip()
    for g in ("✅", "⚠️", "❌"):
        if s.startswith(g):
            s = s[len(g):].lstrip()
            break
    return _bound(s)
```
Banner becomes e.g. `已恢复 148 轮完整对话（model_responses_335438.txt）\n(已写入 backend.history，可直接继续)` — no icon. (Alternative: have the UI own the banner via a new i18n key and ignore the Python string entirely; D1's approach reuses the existing string minus the glyph, which is lower-blast-radius.)

### D3 — make `/scheduler` enumerate the REAL reflect modes + cron tasks
Two parts.
(a) **Data source**: the bridge gains a `Command{name:"scheduler", args:"list"}` reply (additive frame) OR — simpler and matching v4's "UI reads disk for pickers" pattern (cf. `continue_picker::list_sessions`) — a Rust-side reader. Given v4 already reads `temp/model_responses` directly, add `components/scheduler.rs::discover_tasks(repo_root)`:
```rust
/// Discover the REAL reflect modes + cron tasks (slash_cmds.py parity):
///   reflect/*.py (non-`_`)  → mode rows (cadence = "reflect", running via cmdline probe)
///   sche_tasks/*.json       → cron rows (cadence = repeat label, running = enabled flag)
pub fn discover_tasks(repo_root: &Path) -> Vec<ReflectTask> {
    let mut out = Vec::new();
    let mut id = 0usize;
    if let Ok(rd) = std::fs::read_dir(repo_root.join("reflect")) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.ends_with(".py") && !n.starts_with('_') && n != "scheduler.py" {
                out.push(ReflectTask::new(id, n.trim_end_matches(".py").to_string(),
                                          "reflect".into(), false));
                id += 1;
            }
        }
    }
    if let Ok(rd) = std::fs::read_dir(repo_root.join("sche_tasks")) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.ends_with(".json") {
                let (cadence, enabled) = read_cron_meta(&e.path()); // repeat label + enabled
                out.push(ReflectTask::new(id, n.trim_end_matches(".json").to_string(),
                                          cadence, enabled));
                id += 1;
            }
        }
    }
    out
}
```
`read_cron_meta` parses `{repeat|cron|every, enabled, schedule}` (mirror `slash_cmds.list_scheduler_tasks` `:518-522`). Replace the `open_ui_command` "scheduler" path to build `Scheduler::new(discover_tasks(&app.repo_root))` instead of `default_tasks()`. Keep `default_tasks()` ONLY as the empty-dir fallback (degrade-gracefully, scheduler.rs:219-230 doc).
(b) **Apply wiring**: `apply` already computes `to_start`/`to_stop` (scheduler.rs:174-189). Forward them to the bridge as a structured command so it can `slash_cmds.start_service` / stop. Add to `protocol.rs` no new variant needed — reuse `Command{name:"scheduler", args:"start <name>;stop <name>"}` and handle in `ga_bridge.py` by importing `slash_cmds` (it's already on the path) and calling `start_service`/the stop equivalent. The cadence vocabulary surfaced in the panel must be the legal set (`once/daily/weekday/weekly/monthly/every_*`) — render the raw `repeat` label; do not invent `09:00`-style strings.

### D4 — default mouse capture OFF; drop `/mouse`
- `main.rs:2273` init: change `EnterAlternateScreen, EnableMouseCapture` → `EnterAlternateScreen` (no capture), and set `AppState.mouse_capture = false` default. Native drag-select-to-copy now works out of the box (Q10).
- Keep the `Ctrl+Shift+M` keychord (`main.rs:553-559`) as the opt-in to RE-enable wheel scroll for users who want it; its notice strings (`mouse.on`/`mouse.off`, i18n:527-528/791-792) already explain the tradeoff — good.
- Remove the `/mouse` row from `registry.rs:83` and from the `all` test list (`registry.rs:279`) + decrement the `assert_eq!(COMMANDS.len(), all.len())` expectation (`:291`); the `>= 33` floor (`:292`) still holds (39 ≥ 33).

### D5 — cache the `@` file index (kills F5 lag + the F6 "freeze")
Add a memoized snapshot on `AppState` so the walk runs once, not 3×/frame. v4 bans async, so this is a plain cached `Vec` with coarse TTL invalidation (CC-style debounce, not Codex-style actor):
```rust
// in AppState
file_index: std::cell::RefCell<Option<(std::time::Instant, std::sync::Arc<Vec<String>>)>>,

/// Cached project-file list for the `@` picker. Walks at most once per
/// FILE_INDEX_TTL; returns a cheap Arc clone otherwise. (Async is banned;
/// this is the synchronous-equivalent of Codex's FileSearchManager.)
pub fn list_project_files(&self) -> std::sync::Arc<Vec<String>> {
    const TTL: std::time::Duration = std::time::Duration::from_secs(5);
    let mut slot = self.file_index.borrow_mut();
    if let Some((at, files)) = slot.as_ref() {
        if at.elapsed() < TTL { return files.clone(); }
    }
    let files = std::sync::Arc::new(crate::input::paths::list_project_files(&self.repo_root));
    *slot = Some((std::time::Instant::now(), files.clone()));
    files
}
```
The three call sites (`components/mod.rs:706,724`, `main.rs:1525`) keep calling `app.list_project_files()` unchanged — now they share one cached `Arc` per 5s window. The first `@` keystroke pays one walk; subsequent frames/keystrokes are O(1). If `&self` borrow-checking is awkward in the render plane, hoist the walk to once per `@`-session in the keystroke handler and stash `Arc<Vec<String>>` on the composer (`Composer.at_files`), reading it in render. Either removes the per-frame walk. Further (optional) win: have `paths::list_project_files` skip `temp/` and `target/` explicitly (the `DEFAULT_SKIP` set in `paths.rs:15` already skips `target`; add `temp` for this repo's giant log tree).

### D6 — verify Ctrl+S after D5 (no separate fix expected)
After D5, re-run the Ctrl+S → dashboard → describe-task flow. If a residual stall remains, instrument `snapshot_active_into_map` (`app/mod.rs:1325`) — it clones the active transcript; for a 148-turn restored session that `Vec<Line>` clone could be the cost. Fix would be to move the transcript behind an `Arc` (clone the pointer, not the lines). Flag as *measure first*; do not pre-optimize.

### D7 — alias presentation in `/help` + de-dup guard
Add an optional `alias_of: Option<&'static str>` to `SlashCommand` (default `None`) and set it on `sessions`(→status), `abort`(→stop), `tools`/`trace`(→verbose), `exit`(→quit). `/help` then renders aliases as a dimmed `"/tools — alias of /verbose"` under their primary, instead of a full peer row. `palette_matches` (`registry.rs:124`) is unchanged (aliases still complete), but the help GROUPING (`help.group.*`) skips alias rows from the primary list. Cheap, keeps single-source. The existing `registry_resolves_all_commands` test still passes (aliases still resolve).

---

## Review-principle violations (cite principle # + file:line)

- **P2 (局部可推理) + P14 (let-it-crash by blast radius)** — `continue_cmd.restore` returning a human string with embedded glyphs that the bridge forwards blind (`ga_bridge.py:749`) couples the *display* to a *core* string. You cannot reason about "does /continue show bubbles?" by reading the v4 handler (`main.rs:1144`) alone — the answer is three files away in Python. **F1/F2.**
- **P5 (复杂度线性) + P13 (为未来接入设计)** — `scheduler.rs:223-230` hardcodes fake data instead of reading the existing `reflect/` + `sche_tasks/` sources that `slash_cmds.py` already enumerates. A whole feature (the real reflect modes) is missing because the data port was stubbed and never wired. **F3.**
- **P10 (代码极简) + P15 (篇幅跟功能走)** — `app/mod.rs:1188` looks like a trivial one-liner but hides an unbounded full-tree walk invoked 3×/frame. The cost is invisible at the call site; the principle wants cost to be locally legible. **F5.**
- **P7 (可观测) / P3 (可组合)** — `/scheduler` apply (`scheduler.rs:165-189`) computes `to_start`/`to_stop` but `open_ui_command`'s scheduler path (`main.rs:1369`) forwards a bare `Command{name:"scheduler"}` with no diff payload, so the carefully-computed diff is dropped before it reaches the core. The pure state machine is good; the wiring discards its output. **F3(b).**
- **P8 (一致且不意外)** — gerunds English-only (`flavor/mod.rs:153`) while tips are bilingual (`i18n/mod.rs:201/220`): two adjacent "flavor" features handle i18n differently. A zh user gets translated tips but English `Sleuthing…`. **F9.**
- **P6 (约束写进代码)** — `/mouse` default-on (`main.rs:2273`) encodes the *opposite* of the desired invariant ("native selection works by default"); the constraint lives only in a `/mouse` toggle + a keychord the user must discover. Make OFF the typed default. **F4.**
- **P1 (模块边界)** — positive note: `registry.rs` IS a clean single source (the Codex/strum-equivalent); the alias issue (F7) is presentation, not a boundary violation. No change to the resolution core needed.

---

## Open questions / risks

1. **Replay volume (F1/D1)**: a 148-turn session emits ~296 `MessageBegin/Delta/End` triples through the bridge pipe on restore. Confirm the v4 transcript ingest + wrap-cache handles a burst of ~300 messages without a visible stall (it should — they're plain frames — but the wrap cache invalidation on each `MessageEnd` is the thing to watch). Fallback: replay only the last N (e.g. 40) bubbles + a dim "…earlier turns in backend.history" line. User wants ALL, so default all; cap only if measured.
2. **`extract_ui_messages` content shape**: it embeds `**LLM Running (Turn N) ...**` markers and 5-backtick tool fences (`continue_cmd.py:543,476`). v3 commits these as-is and its fold logic collapses them. Confirm v4's markdown/fold path (the tool-chip folding) recognizes the same string shape; if not, the replayed assistant bubbles will show raw `**LLM Running…**` text. (v3 works, so the shape is proven — just verify v4's fold parity.)
3. **`/scheduler` start/stop side effects (D3b)**: `slash_cmds.start_service` spawns detached processes (`slash_cmds.py:361-373`) and `find_running` does a ~30ms process scan. Running this from the bridge is fine, but confirm the bridge process has the same cwd/Python as v2 expects (`_ROOT` in slash_cmds resolves from its own `__file__`, so OK). Risk: stopping a reflect task — `slash_cmds` has `start_service` but the "stop" verb needs verification (the v2 picker toggles via re-running/`find_running`; confirm there's a kill path or it's "untick = don't start", in which case stop = no-op for already-running detached procs). This affects whether the v4 stop-diff is actionable or advisory.
4. **Mouse default (F4)**: turning capture off loses wheel-scroll. Confirm keyboard scroll (PgUp/PgDn/Home/End, Ctrl+wheel-equivalent) fully covers transcript navigation so no one is stranded. Q10 is explicit, so proceed, but the keychord (`Ctrl+Shift+M`) must remain discoverable (it's in the mouse.on/off notice and should be in `/help`).
5. **File-index TTL (D5)**: 5s TTL means a newly-created file isn't `@`-completable for ≤5s. Acceptable for chat. If a user complains, drop to invalidation-on-submit. Also verify `RefCell` interior mutability is OK in the render plane (render takes `&AppState`); if borrow conflicts, use the composer-stash variant in D5.
6. **i18n test gate**: every new key (D2 banner if UI-owned, D3 ~12 scheduler/reflect keys, D7 alias label) MUST be added to BOTH `EN_PAIRS` and `ZH_PAIRS` or `dictionaries_cover_the_same_keys` (`i18n/mod.rs:843`) fails the build. The gerund bilingual pool (F9) needs a parallel `GERUNDS_ZH` with EQUAL length or a length-parity test will be the right guard to add.

---

## Appendix — i18n gaps + tip/egg additions (zh+en)

### i18n gaps (keys to add to BOTH `EN_PAIRS` and `ZH_PAIRS`, `i18n/mod.rs`)
For the F3 real-scheduler panel (reflect modes vs cron tasks split):
- `scheduler.kind.reflect` = `reflect mode` / `反射模式`
- `scheduler.kind.cron` = `cron task` / `定时任务`
- `scheduler.cadence.reflect` = `watcher` / `监控守护`
- `scheduler.repeat.once/daily/weekday/weekly/monthly` = `once/daily/weekdays/weekly/monthly` / `一次/每日/工作日/每周/每月`
- `scheduler.empty` = `no reflect modes or cron tasks found` / `未找到反射模式或定时任务`
For F2/F1 (restore banner, if moved UI-side instead of D2's Python strip):
- `continue.restored` = `restored {n} turns` / `已恢复 {n} 轮对话` (NO ✅)
- `continue.replaying` = `replaying conversation…` / `正在重放历史对话…`
For F4 help (the `/mouse`-less description of native selection):
- `mouse.hint.native` = `drag to select & copy natively; Ctrl+Shift+M re-enables wheel scroll` / `拖动即可原生选中复制；Ctrl+Shift+M 重新开启滚轮滚动`
For F7:
- `help.alias_of` = `alias of` / `等同于`

### tips additions (append to `TIPS_EN` AND `TIPS_ZH`, keep lengths equal — `i18n/mod.rs:201,220`)
- EN: `Tip: /scheduler lists every reflect mode (reflect/*.py) and cron task (sche_tasks/*.json) — tick to start, untick to stop.`
  ZH: `Tip: /scheduler 列出所有反射模式（reflect/*.py）与定时任务（sche_tasks/*.json）—— 勾选启动，取消勾选停止。`
- EN: `Tip: mouse capture is off by default — drag to select & copy natively; Ctrl+Shift+M toggles wheel scroll.`
  ZH: `Tip: 默认关闭鼠标捕获 —— 拖动即可原生选中复制；Ctrl+Shift+M 切换滚轮滚动。`
- EN: `Tip: after /continue the full prior conversation replays into the transcript, not just a one-line summary.`
  ZH: `Tip: /continue 之后会把完整的历史对话重放进窗口，而不只是一行摘要。`

### gerund/status-word egg additions — make the pool BILINGUAL (F9)
Add a `GERUNDS_ZH: &[&str]` parallel to `GERUNDS` (`flavor/mod.rs:154`), same length, and make `gerund(lang, tick)` pick the pool by `Lang` (signature change: `pub fn gerund(lang: Lang, tick: u64) -> &'static str`). New EN eggs to APPEND (Q12 "继续补充"), each with a zh sibling at the same index:
| EN (append to GERUNDS) | ZH (same index in GERUNDS_ZH) |
|---|---|
| Sleuthing | 探案中 |
| Reticulating | 织网中 |
| Spelunking | 探洞中 |
| Conjuring | 施法中 |
| Marinating | 腌制中 |
| Percolating | 渗滤中 |
| Untangling | 解结中 |
| Bamboozling | 谋划中 |
| Galaxy-braining | 烧脑中 |
| Vibing | 找感觉中 |
| Summoning daemons | 召唤守护进程 |
| Bikeshedding | 纠结细节中 |
| Yak-shaving | 剃牦牛中 |
(For the EXISTING 34 EN gerunds, fill `GERUNDS_ZH` with translations: Pondering→沉思中, Brewing→酝酿中, Compiling→编译中, Distilling→提炼中, Calibrating→校准中, Synthesizing→合成中, Routing→路由中, Threading→穿线中, Caching→缓存中, Streaming→流式中, Resolving→解析中, Tuning→调优中, …). Add a `gerunds_parity` test: `assert_eq!(GERUNDS.len(), GERUNDS_ZH.len())` — the build-time guard that keeps the two in lockstep.)
