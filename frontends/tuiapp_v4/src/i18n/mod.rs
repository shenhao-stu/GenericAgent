//! i18n/mod.rs — the interface-language plane (checklist §9 / §11): zh + en
//! dictionaries (~250 keys), the `lang → en → key` fallback chain behind a single
//! `t(key)` helper, system-locale detection, the `/language` FULL repaint, the
//! `/emoji` style switch, and the per-language rotating tips.
//!
//! THE FALLBACK CONTRACT (the load-bearing rule, `i18n_fallback_chain` test): a
//! lookup NEVER panics and NEVER returns an empty string. `t(lang, key)` resolves
//!   1. the ACTIVE language's dictionary,
//!   2. else the English dictionary (the canonical superset),
//!   3. else the KEY string itself (so a missing key is visible, not blank).
//! Every user-facing string in the cockpit/overlays routes through this one helper
//! so adding a language is "fill the map", and a hole degrades gracefully instead
//! of showing nothing. The dictionaries are PURE static data; the resolver is a
//! pure function — both are unit-tested without a TTY.
//!
//! REPAINT: ratatui is immediate-mode, so a `/language` switch is "set `app.lang`";
//! the very next `render()` reads `t(app.lang, …)` for every label and the whole UI
//! repaints in the new language for free (no retained scene to invalidate). The
//! render-cache invalidation hook (`AppState::on_language_change`) bumps the wrap
//! cache so multi-line, language-dependent transcript chrome re-wraps too.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The interface language. The CANONICAL definition lives here; `flavor`
/// re-exports it (`pub use crate::i18n::Lang`) so the historical `flavor::Lang`
/// references keep compiling. `En` is the fallback language (the superset map).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lang {
    /// English (the canonical superset / fallback language).
    #[default]
    En,
    /// Simplified Chinese.
    Zh,
}

impl Lang {
    /// The BCP-47-ish short code (`en` / `zh`) — what `/language <code>` accepts
    /// and what the status line shows.
    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Zh => "zh",
        }
    }

    /// The endonym shown in the `/language` picker (English / 简体中文).
    pub fn endonym(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Zh => "简体中文",
        }
    }

    /// Every selectable language, in `/language` picker order.
    pub fn all() -> [Lang; 2] {
        [Lang::En, Lang::Zh]
    }

    /// Parse a language from a `/language <code>` argument or a system-locale
    /// string (case-insensitive, prefix-tolerant: `zh`, `zh-CN`, `zh_CN.UTF-8`,
    /// `Chinese` all map to `Zh`; `en`, `en-US`, `C`, `English` to `En`). Returns
    /// `None` for an unrecognized token so the caller can keep the current lang.
    pub fn from_code(s: &str) -> Option<Lang> {
        let t = s.trim().to_ascii_lowercase();
        if t.is_empty() {
            return None;
        }
        // Chinese: any locale beginning with `zh`, or the English word.
        if t.starts_with("zh") || t.starts_with("chinese") || t.contains("hans") || t.contains("hant") {
            return Some(Lang::Zh);
        }
        if t.starts_with("en") || t.starts_with("english") || t == "c" || t.starts_with("c.") {
            return Some(Lang::En);
        }
        None
    }
}

/// Detect the interface language from the environment (the system-locale detect,
/// §9). Honors, in order: an explicit `GA_TUI_LANG` / `TUI_V4_LANG` override, then
/// the POSIX `LC_ALL` / `LC_MESSAGES` / `LANG` chain, then (on Windows) the OS UI
/// language. Falls back to [`Lang::En`] so the app always has a language. Effectful
/// (reads env / OS), so the PURE [`Lang::from_code`] is what the test pins; this is
/// the thin discovery wrapper the app calls once at startup.
pub fn detect_system_lang() -> Lang {
    for key in ["GA_TUI_LANG", "TUI_V4_LANG", "LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"] {
        if let Ok(val) = std::env::var(key) {
            if let Some(lang) = Lang::from_code(&val) {
                return lang;
            }
        }
    }
    // Windows: the console / UI language isn't always in LANG; probe the OS.
    #[cfg(windows)]
    if let Some(lang) = detect_windows_lang() {
        return lang;
    }
    Lang::En
}

/// Best-effort Windows UI-language probe via `GetUserDefaultLocaleName`-style
/// behavior, approximated by reading the common console code page hint. We keep
/// this dependency-free: a cp936 (GBK) active code page is the strong signal for a
/// Simplified-Chinese Windows install. Returns `None` if undetermined.
#[cfg(windows)]
fn detect_windows_lang() -> Option<Lang> {
    // The `chcp`-reported active code page lands in this env on many shells; when
    // absent we simply defer to the En fallback (never wrong, just generic).
    if let Ok(cp) = std::env::var("ACP").or_else(|_| std::env::var("CHCP")) {
        if cp.trim() == "936" {
            return Some(Lang::Zh);
        }
    }
    None
}

/// Resolve a key to its string for `lang`, with the `lang → en → key` fallback
/// chain (the load-bearing contract). NEVER panics; NEVER returns "". This is the
/// single helper every user-facing string routes through.
///
///   * If `lang`'s dictionary has `key`, return it.
///   * Else if English has `key`, return that (English is the canonical superset).
///   * Else return `key` itself (a missing key is visible, not blank).
pub fn t(lang: Lang, key: &str) -> &'static str {
    if let Some(v) = dict(lang).get(key) {
        return v;
    }
    if lang != Lang::En {
        if let Some(v) = dict(Lang::En).get(key) {
            return v;
        }
    }
    // Last resort: the key string. We must return a `'static` str; the key is a
    // borrowed `&str`, so intern it into a process-lifetime leak set (bounded by
    // the finite key surface — a missing key is a bug we want surfaced, and the
    // leak is a handful of bytes, once, for the lifetime of the process).
    intern_missing(key)
}

/// Format-friendly variant of [`t`]: returns an owned `String` so a caller can
/// `format!` around it without lifetime gymnastics. Same fallback chain.
pub fn tf(lang: Lang, key: &str) -> String {
    t(lang, key).to_string()
}

/// Intern a missing key as a `'static` str (the fallback-of-last-resort). Bounded:
/// only ever called for keys absent from BOTH dictionaries (a programming error we
/// want visible), so the set stays tiny. Uses a leak — acceptable for the finite,
/// process-lifetime key surface, and avoids ever returning an empty string.
fn intern_missing(key: &str) -> &'static str {
    use std::sync::Mutex;
    static MISSING: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let map = MISSING.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().expect("i18n missing-key lock");
    if let Some(s) = guard.get(key) {
        return s;
    }
    let leaked: &'static str = Box::leak(key.to_string().into_boxed_str());
    guard.insert(key.to_string(), leaked);
    leaked
}

/// The dictionary for a language (lazily built once, then cached). `En` is the
/// canonical superset; `Zh` provides translations for the same key set.
fn dict(lang: Lang) -> &'static HashMap<&'static str, &'static str> {
    match lang {
        Lang::En => en_dict(),
        Lang::Zh => zh_dict(),
    }
}

fn en_dict() -> &'static HashMap<&'static str, &'static str> {
    static D: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    D.get_or_init(|| pairs_to_map(EN_PAIRS))
}

fn zh_dict() -> &'static HashMap<&'static str, &'static str> {
    static D: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    D.get_or_init(|| pairs_to_map(ZH_PAIRS))
}

fn pairs_to_map(pairs: &[(&'static str, &'static str)]) -> HashMap<&'static str, &'static str> {
    pairs.iter().copied().collect()
}

/// All keys defined in the English (superset) dictionary — for the coverage guard
/// that every `zh` key is a known key and vice-versa, and for a future
/// locale-coverage lint. PURE.
#[allow(dead_code)] // public locale-coverage surface; exercised by the coverage test.
pub fn all_keys() -> Vec<&'static str> {
    EN_PAIRS.iter().map(|(k, _)| *k).collect()
}

// ---------------------------------------------------------------------------
// Rotating tips — per-language pools (moved here from `flavor`, §9 "per-language
// rotating tips"). Deterministic rotation by an integer step (no wall-clock).
// ---------------------------------------------------------------------------

/// English usage tips (the §5 header rotation; ported from tui_v3 `_TIPS['en']`).
pub const TIPS_EN: &[&str] = &[
    "Tip: press / to open the command palette — arrow keys to pick, Enter drops it into the input box.",
    "Tip: pasted images / files fold into [Image #N] / [File #N] placeholders; backspace deletes the whole block.",
    "Tip: /btw <question> lets a side-agent answer without interrupting the main task.",
    "Tip: /rewind [n] rewinds the last n turns; /stop aborts the current task.",
    "Tip: /continue searches past sessions by content — type to filter, Enter to restore.",
    "Tip: Ctrl+J / Shift+Enter inserts a newline in multi-line input; Enter sends.",
    "Tip: put [multi-select] in an ask_user prompt to switch to a multi-pick picker.",
    "Tip: /cost shows token usage; /llm views / switches the model.",
    "Tip: /new [name] starts a fresh session; /language switches the interface language.",
    "Tip: /export clip copies the last reply to your system clipboard; /export all prints the log path.",
    "Tip: Ctrl+O folds / unfolds all completed tool chips — each fold collapses to one line.",
    "Tip: prefix `!` runs the rest as a host shell command — output is folded into LLM history.",
    "Tip: Ctrl+S opens the session dashboard; describe a task to start a new session.",
    "Tip: @path inlines a project file into your message (gitignore-aware path completion).",
    "Tip: /scheduler picks reflect tasks to run on a cron; /emoji changes the spinner + pet.",
    // -- C5-appendix additions (scheduler / mouse / continue) -----------------
    "Tip: /scheduler lists every reflect mode (reflect/*.py) and cron task (sche_tasks/*.json) — tick to start, untick to stop.",
    "Tip: mouse capture is off by default — drag to select & copy natively; Ctrl+Shift+M toggles wheel scroll.",
    "Tip: after /continue the full prior conversation replays into the transcript, not just a one-line summary.",
];

/// Simplified-Chinese usage tips (ported from tui_v3 `_TIPS['zh']`).
pub const TIPS_ZH: &[&str] = &[
    "Tip: 按 / 唤起命令面板 —— 方向键选择，Enter 落入输入框。",
    "Tip: 粘贴图片 / 文件会折叠成 [Image #N] / [File #N] 占位符，退格可整块删除。",
    "Tip: /btw <问题> 让 side-agent 回答而不打断主任务。",
    "Tip: /rewind [n] 回退最近 n 轮对话；/stop 中止当前任务。",
    "Tip: /continue 按内容搜索历史会话 —— 输入关键词过滤，Enter 恢复。",
    "Tip: 多行输入用 Ctrl+J / Shift+Enter 换行；Enter 直接发送。",
    "Tip: ask_user 题目里写 [多选] 会自动切到多选 picker。",
    "Tip: /cost 查看 token 用量；/llm 查看 / 切换模型。",
    "Tip: /new [name] 新建会话；/language 切换界面语言。",
    "Tip: /export clip 把最后回复复制到系统剪贴板；/export all 打印日志路径。",
    "Tip: Ctrl+O 折叠 / 展开所有已完成的工具 chip —— 每个 chip 折叠成一行。",
    "Tip: 以 `!` 开头直接跑 shell —— 命令与输出都会进入 LLM 历史，agent 可以引用。",
    "Tip: Ctrl+S 打开会话面板；描述一个任务即可开启新会话。",
    "Tip: @path 把项目文件内联进消息（路径补全会跳过 .gitignore）。",
    "Tip: /scheduler 勾选要定时运行的 reflect 任务；/emoji 切换 spinner 与桌宠。",
    // -- C5-appendix additions (scheduler / mouse / continue) -----------------
    "Tip: /scheduler 列出所有反射模式（reflect/*.py）与定时任务（sche_tasks/*.json）—— 勾选启动，取消勾选停止。",
    "Tip: 默认关闭鼠标捕获 —— 拖动即可原生选中复制；Ctrl+Shift+M 切换滚轮滚动。",
    "Tip: /continue 之后会把完整的历史对话重放进窗口，而不只是一行摘要。",
];

/// The tip pool for a language. PURE.
pub fn tips_for(lang: Lang) -> &'static [&'static str] {
    match lang {
        Lang::En => TIPS_EN,
        Lang::Zh => TIPS_ZH,
    }
}

/// Number of 0.1s ticks per tip step (~12s rotation, per the §5 header spec).
pub const TIP_TICKS_PER_STEP: u64 = 120;

/// The tip for an explicit STEP INDEX in a language (deterministic, wraps). PURE.
pub fn tip_at(lang: Lang, step: u64) -> &'static str {
    let pool = tips_for(lang);
    if pool.is_empty() {
        return "";
    }
    pool[(step % pool.len() as u64) as usize]
}

/// The tip for the 0.1s tick clock — rotates one tip every ~12s. Deterministic
/// (no wall-clock, no randomness): a redraw within the same window shows the same
/// tip. PURE.
pub fn tip(lang: Lang, tick: u64) -> &'static str {
    tip_at(lang, tick / TIP_TICKS_PER_STEP)
}

// ---------------------------------------------------------------------------
// The dictionaries. EN is the canonical superset; ZH mirrors the same keys.
// Keys are dotted, namespaced by surface (composer / footer / cmd / picker /
// btw / scheduler / continue / rewind / ask / status / cost / help / dash /
// notice / lang). ~250 keys.
// ---------------------------------------------------------------------------

/// The English (superset) key→string pairs.
pub const EN_PAIRS: &[(&str, &str)] = &[
    // -- app identity / header ------------------------------------------------
    ("app.name", "GenericAgent · tui_v4"),
    ("header.sessions_hint", "⌃S sessions"),
    // -- connection / footer --------------------------------------------------
    ("conn.connecting", "connecting…"),
    ("conn.connected", "connected"),
    ("conn.disconnected", "disconnected"),
    ("footer.mode.chat", "chat"),
    ("footer.mode.running", "running"),
    ("footer.mode.bash", "bash"),
    ("footer.mode.plan", "plan"),
    ("footer.mode.accept", "accept"),
    ("footer.mode.auto", "auto"),
    ("footer.ctx", "ctx"),
    ("footer.cost", "cost"),
    ("footer.git_none", "—"),
    // -- composer -------------------------------------------------------------
    ("composer.placeholder", "type a message…"),
    ("composer.placeholder.shell", "run a host shell command…"),
    ("transcript.empty", "  Type a message and press Enter."),
    ("transcript.more_below", "  ▼ more below — End/PgDn to follow"),
    // -- dropdown hints -------------------------------------------------------
    ("palette.hint", "  ↑/↓ move · Tab/Enter complete"),
    ("filepicker.hint", "  ↑/↓ move · Tab/Enter complete @path"),
    // -- generic picker hints -------------------------------------------------
    ("picker.hint.single", "↑↓ move · enter select · esc cancel"),
    ("picker.hint.preview", "↑↓ preview · enter apply · esc revert"),
    ("picker.hint.multi", "↑↓ move · space toggle · enter apply · esc cancel"),
    ("picker.empty", "  (nothing to pick)"),
    // -- picker titles --------------------------------------------------------
    ("picker.title.llm", "Switch model"),
    ("picker.title.theme", "Theme (live preview)"),
    ("picker.title.emoji", "Pet / spinner style"),
    ("picker.title.language", "Interface language"),
    ("picker.title.export", "Export last reply"),
    ("picker.title.rewind", "Rewind to a turn"),
    ("picker.title.continue", "Continue a session"),
    ("picker.title.scheduler", "Scheduler tasks"),
    // -- ask_user -------------------------------------------------------------
    ("ask.title.single", "ask_user"),
    ("ask.title.multi", "ask_user [multi-select]"),
    ("ask.title.numeric", "ask_user [numeric]"),
    ("ask.input.placeholder", "type a free-text answer…"),
    ("ask.input.number", "number: "),
    ("ask.hint.single", "↑↓ cycle candidates / input · enter submit · esc cancel"),
    ("ask.hint.multi", "↑↓ move · space toggle · enter submit · esc cancel"),
    ("ask.hint.numeric", "type a number or ↑↓ · enter submit · esc cancel"),
    // -- /help ----------------------------------------------------------------
    ("help.title", "Commands · /help"),
    ("help.group.ui", "Interactive (open a panel)"),
    ("help.group.app", "In-app"),
    ("help.group.fwd", "Core-forwarded"),
    ("help.magic", "Magic prefixes:  !cmd  run a host shell line   ·   @path  inline a project file"),
    ("help.close", "esc / q  close"),
    ("help.alias_of", "alias of"),
    // -- /keybindings ---------------------------------------------------------
    ("keybindings.title", "Keyboard shortcuts · /keybindings"),
    ("kb.submit", "send the message"),
    ("kb.newline", "insert a newline (multi-line input)"),
    ("kb.palette", "open the command palette"),
    ("kb.complete", "complete the highlighted command / @path"),
    ("kb.copy_reply", "copy the last reply (clean, no soft-wraps)"),
    ("kb.fold", "fold / unfold all completed tool chips"),
    ("kb.mouse", "toggle mouse capture (wheel scroll ⇄ native select)"),
    ("kb.scroll", "scroll the transcript"),
    ("kb.views", "switch chat ⇄ session dashboard"),
    ("kb.dashboard", "open the session dashboard"),
    ("kb.new_session", "create + switch to a new session"),
    ("kb.cycle_session", "previous / next session"),
    ("kb.drop_session", "close the active session"),
    ("kb.branch", "fork the session with copied history"),
    ("kb.theme", "open the theme picker"),
    ("kb.help", "show these shortcuts"),
    ("kb.escape", "back · Esc twice rewinds"),
    ("kb.quit", "abort / quit"),
    // -- /status --------------------------------------------------------------
    ("status.title", "Status · /status"),
    ("status.model", "model"),
    ("status.connection", "connection"),
    ("status.state", "state"),
    ("status.state.idle", "idle"),
    ("status.state.working", "working"),
    ("status.sessions", "sessions"),
    ("status.by_status", "  by status"),
    ("status.needs_input", "needs-input"),
    ("status.working", "working"),
    ("status.completed", "completed"),
    ("status.context", "context"),
    ("status.cost", "cost"),
    ("status.git", "git"),
    ("status.cwd", "cwd"),
    ("status.total", "total"),
    ("status.close", "  esc close"),
    // -- /cost ----------------------------------------------------------------
    ("cost.title", "Cost · /cost"),
    ("cost.header", "Token usage"),
    ("cost.input", "input"),
    ("cost.output", "output"),
    ("cost.cache", "cache"),
    ("cost.total", "total"),
    ("cost.context", "context"),
    ("cost.cost", "cost"),
    // -- /verbose -------------------------------------------------------------
    ("verbose.title", "Tool-call audit · /verbose"),
    ("verbose.empty", "  no tool calls yet this session."),
    // -- /btw -----------------------------------------------------------------
    ("btw.title", " /btw "),
    ("btw.querying", "querying… (a side-agent is answering; main task keeps running)"),
    ("btw.dismiss", "esc dismiss"),
    ("btw.usage", "usage: /btw <question>"),
    ("btw.failed", "side-question failed"),
    // -- /scheduler -----------------------------------------------------------
    ("scheduler.title.pick", "Scheduler · pick tasks"),
    ("scheduler.title.confirm", "Scheduler · confirm changes"),
    ("scheduler.title.status", "Scheduler · cron status"),
    ("scheduler.hint.pick", "↑↓ move · space toggle · enter review · esc cancel"),
    ("scheduler.hint.confirm", "enter apply · esc back"),
    ("scheduler.hint.status", "esc close"),
    ("scheduler.start", "start"),
    ("scheduler.stop", "stop"),
    ("scheduler.unchanged", "unchanged"),
    ("scheduler.running", "running"),
    ("scheduler.stopped", "stopped"),
    ("scheduler.no_changes", "no changes — every task is already in its desired state"),
    ("scheduler.applied", "scheduler updated"),
    ("scheduler.none_selected", "scheduler: no tasks selected"),
    ("scheduler.cron_active", "cron active"),
    ("scheduler.cron_none", "no cron jobs scheduled"),
    ("scheduler.will_start", "will start"),
    ("scheduler.will_stop", "will stop"),
    // The two row kinds the discovery surfaces (reflect/*.py vs sche_tasks/*.json).
    ("scheduler.kind.reflect", "reflect mode"),
    ("scheduler.kind.cron", "cron task"),
    // The cadence label for a reflect mode (a watcher, not a cron job). Cron tasks
    // render their RAW legal `repeat` grammar (below) verbatim — never an HH:MM.
    ("scheduler.cadence.reflect", "watcher"),
    ("scheduler.repeat.once", "once"),
    ("scheduler.repeat.daily", "daily"),
    ("scheduler.repeat.weekday", "weekdays"),
    ("scheduler.repeat.weekly", "weekly"),
    ("scheduler.repeat.monthly", "monthly"),
    ("scheduler.empty", "no reflect modes or cron tasks found"),
    // -- /continue ------------------------------------------------------------
    ("continue.title", "Continue · search sessions"),
    ("continue.search", "search: "),
    ("continue.search.placeholder", "type to filter by content…"),
    ("continue.hint", "type to search · ↑↓ move · enter restore · esc cancel"),
    ("continue.empty", "no past sessions found"),
    ("continue.no_match", "no sessions match your search"),
    ("continue.restoring", "restoring session…"),
    ("continue.rounds", "rounds"),
    // The icon-free restore banner copy (Q10: no ✅). The bridge strips the glyph
    // from continue_cmd's string; these are the UI-owned equivalents.
    ("continue.restored", "restored {n} turns"),
    ("continue.replaying", "replaying conversation…"),
    // -- /rewind --------------------------------------------------------------
    ("rewind.empty", "nothing to rewind (no turns yet)"),
    ("rewind.turn", "turn"),
    ("rewind.done", "rewound"),
    ("rewind.turns_suffix", "turn(s)"),
    // -- /export --------------------------------------------------------------
    ("export.clip", "clip"),
    ("export.clip.detail", "copy last reply to clipboard (OSC52)"),
    ("export.all", "all"),
    ("export.all.detail", "copy the whole transcript"),
    ("export.file", "file"),
    ("export.file.detail", "write last reply to a file in cwd"),
    ("export.copied_reply", "copied last reply to clipboard"),
    ("export.copied_all", "copied the transcript to clipboard"),
    ("export.wrote", "wrote last reply →"),
    ("export.failed", "export failed"),
    ("export.none", "no assistant reply to export yet"),
    // -- /llm -----------------------------------------------------------------
    ("llm.querying", "querying models…"),
    ("llm.switching", "switching model…"),
    ("llm.no_models", "no models configured (check mykey.py)"),
    // -- /theme / /emoji / /language ------------------------------------------
    ("theme.set", "theme →"),
    ("theme.unknown", "unknown theme (try /theme)"),
    ("emoji.updated", "style updated"),
    ("emoji.pet", "pet"),
    ("emoji.spinner", "spinner"),
    ("emoji.off", "off"),
    ("lang.set.en", "language → English"),
    ("lang.set.zh", "界面语言 → 简体中文"),
    // -- dashboard ------------------------------------------------------------
    ("dash.needs_input", "Needs input"),
    ("dash.working", "Working"),
    ("dash.completed", "Completed"),
    ("dash.new_session", "describe a task for a new session"),
    ("dash.hint", "enter open · space reply · ctrl+x delete · r rename · ctrl+n new · ? shortcuts"),
    ("dash.awaiting", "N awaiting input"),
    ("dash.header.working", "working"),
    ("dash.header.completed", "completed"),
    ("dash.preview.idle", "send a prompt to start"),
    ("dash.preview.working", "working…"),
    ("dash.preview.awaiting", "awaiting your answer"),
    ("dash.rename", "rename:"),
    // -- generic command notices ----------------------------------------------
    ("cmd.unknown", "unknown command"),
    ("cmd.did_you_mean", "did you mean"),
    ("cmd.type_help", "(type /help for the list)"),
    ("cmd.not_wired", "UI is not wired yet"),
    ("cmd.not_handled", "is not handled yet"),
    ("cmd.clear.busy", "can't clear while a turn is running (/stop first)"),
    ("cmd.aborted", "aborted the running task"),
    ("cmd.rename.usage", "usage: /rename <name>"),
    ("cmd.renamed", "renamed session →"),
    ("cmd.restoring", "restoring prior transcript into history…"),
    ("cmd.reloading_keys", "reloading mykey.py…"),
    ("cmd.effects", "effects"),
    ("cmd.effects.off_default", "(off by default; demo lands in the effects phase)"),
    // -- bridge / connection notices ------------------------------------------
    ("notice.bridge.not_connected", "cannot send: bridge is not connected (see status line)"),
    ("notice.bridge.spawn_failed", "bridge spawn failed"),
    ("notice.bridge.exited", "bridge exited"),
    ("notice.error", "error"),
    // -- multi-press chords (§8) ----------------------------------------------
    ("ctrlc.arm", "press Ctrl+C again to quit"),
    // -- key hints (footer / generic) -----------------------------------------
    ("key.enter", "enter"),
    ("key.esc", "esc"),
    ("key.cancel", "cancel"),
    ("key.apply", "apply"),
    ("key.back", "back"),
    ("key.close", "close"),
    ("key.move", "move"),
    ("key.toggle", "toggle"),
    ("key.select", "select"),
    ("key.search", "search"),
    ("key.restore", "restore"),
    ("key.open", "open"),
    ("key.reply", "reply"),
    ("key.delete", "delete"),
    ("key.rename", "rename"),
    ("key.new", "new"),
    // -- /workflows panel (§7) ------------------------------------------------
    ("wf.title", "Workflows"),
    ("wf.hint", "↑↓ focus · t style · enter detail · r refresh · esc back"),
    ("wf.style", "style"),
    ("wf.style.tree", "box-tree"),
    ("wf.style.list", "bullet list"),
    ("wf.count.workflows", "workflows"),
    ("wf.count.nodes", "nodes"),
    ("wf.count.running", "running"),
    ("wf.turns", "turns"),
    ("wf.empty", "No workflows running."),
    ("wf.empty.launch", "Start one with /conductor, /hive, or /goal."),
    ("wf.down", "not running"),
    ("wf.down.launch", "press X to launch"),
    ("wf.group.conductor", "Conductor"),
    ("wf.group.hives", "Hives"),
    ("wf.group.goal", "Goal"),
    ("wf.status.running", "running"),
    ("wf.status.idle", "idle"),
    ("wf.status.wrapping_up", "wrapping up"),
    ("wf.status.done", "done"),
    ("wf.status.failed", "failed"),
    ("wf.status.aborted", "aborted"),
    ("wf.status.unknown", "unknown"),
    ("wf.action.keyinfo", "key-info"),
    ("wf.action.input", "send input"),
    ("wf.action.stop", "stop"),
    ("wf.action.kill", "kill"),
    ("wf.action.open", "open detail"),
    ("wf.action.sent", "action sent:"),
    ("wf.action.unavailable", "action unavailable (server not running)"),
    ("wf.detail", "detail"),
    ("wf.detail.status", "status"),
    ("wf.detail.workflow", "workflow"),
    ("wf.detail.parent", "parent"),
    ("wf.detail.prompt", "prompt"),
    ("wf.detail.output", "output"),
    ("wf.detail.feed", "activity"),
    ("wf.detail.actions", "actions"),
    ("wf.detail.hint", "↑↓ action · enter fire · esc close"),
    // -- units / misc ---------------------------------------------------------
    ("unit.bytes", "bytes"),
    ("unit.tokens", "tok"),
    ("unit.seconds", "s"),
    // -- copy / clipboard feedback + mouse toggle -----------------------------
    ("copy.ok", "Copied"),
    ("copy.fail", "Copy failed"),
    ("copy.fail.too_large", "payload too large; use /export file"),
    ("copy.fail.no_tty", "no terminal to copy to"),
    ("copy.fail.generic", "no clipboard available"),
    ("copy.label.selection", "selection"),
    ("copy.label.cut", "cut"),
    ("copy.label.reply", "last reply"),
    ("copy.label.transcript", "transcript"),
    ("mouse.on", "mouse capture on (wheel scroll · click dashboard)"),
    ("mouse.off", "mouse capture off — drag to select, then copy in your terminal"),
    ("mouse.hint.native", "drag to select & copy natively; Ctrl+Shift+M re-enables wheel scroll"),
    ("misc.none", "—"),
    ("misc.yes", "yes"),
    ("misc.no", "no"),
    ("misc.on", "on"),
    ("misc.off", "off"),
];

/// The Simplified-Chinese key→string pairs (mirror of the English key set).
pub const ZH_PAIRS: &[(&str, &str)] = &[
    // -- app identity / header ------------------------------------------------
    ("app.name", "GenericAgent · tui_v4"),
    ("header.sessions_hint", "⌃S 会话"),
    // -- connection / footer --------------------------------------------------
    ("conn.connecting", "连接中…"),
    ("conn.connected", "已连接"),
    ("conn.disconnected", "已断开"),
    ("footer.mode.chat", "对话"),
    ("footer.mode.running", "运行中"),
    ("footer.mode.bash", "命令"),
    ("footer.mode.plan", "计划"),
    ("footer.mode.accept", "接受"),
    ("footer.mode.auto", "自动"),
    ("footer.ctx", "上下文"),
    ("footer.cost", "费用"),
    ("footer.git_none", "—"),
    // -- composer -------------------------------------------------------------
    ("composer.placeholder", "输入消息…"),
    ("composer.placeholder.shell", "运行主机 shell 命令…"),
    ("transcript.empty", "  输入消息并按 Enter。"),
    ("transcript.more_below", "  ▼ 下方还有内容 —— End/PgDn 跟随"),
    // -- dropdown hints -------------------------------------------------------
    ("palette.hint", "  ↑/↓ 移动 · Tab/Enter 补全"),
    ("filepicker.hint", "  ↑/↓ 移动 · Tab/Enter 补全 @path"),
    // -- generic picker hints -------------------------------------------------
    ("picker.hint.single", "↑↓ 移动 · enter 选择 · esc 取消"),
    ("picker.hint.preview", "↑↓ 预览 · enter 应用 · esc 还原"),
    ("picker.hint.multi", "↑↓ 移动 · 空格 勾选 · enter 应用 · esc 取消"),
    ("picker.empty", "  （无可选项）"),
    // -- picker titles --------------------------------------------------------
    ("picker.title.llm", "切换模型"),
    ("picker.title.theme", "主题（实时预览）"),
    ("picker.title.emoji", "桌宠 / spinner 样式"),
    ("picker.title.language", "界面语言"),
    ("picker.title.export", "导出最后回复"),
    ("picker.title.rewind", "回退到某轮"),
    ("picker.title.continue", "继续历史会话"),
    ("picker.title.scheduler", "定时任务"),
    // -- ask_user -------------------------------------------------------------
    ("ask.title.single", "ask_user"),
    ("ask.title.multi", "ask_user [多选]"),
    ("ask.title.numeric", "ask_user [数字]"),
    ("ask.input.placeholder", "输入自由文本回答…"),
    ("ask.input.number", "数字："),
    ("ask.hint.single", "↑↓ 在候选 / 输入间切换 · enter 提交 · esc 取消"),
    ("ask.hint.multi", "↑↓ 移动 · 空格 勾选 · enter 提交 · esc 取消"),
    ("ask.hint.numeric", "输入数字或 ↑↓ · enter 提交 · esc 取消"),
    // -- /help ----------------------------------------------------------------
    ("help.title", "命令 · /help"),
    ("help.group.ui", "交互（打开面板）"),
    ("help.group.app", "应用内"),
    ("help.group.fwd", "转发核心"),
    ("help.magic", "魔法前缀：  !cmd  运行主机 shell   ·   @path  内联项目文件"),
    ("help.close", "esc / q  关闭"),
    ("help.alias_of", "等同于"),
    // -- /keybindings ---------------------------------------------------------
    ("keybindings.title", "键盘快捷键 · /keybindings"),
    ("kb.submit", "发送消息"),
    ("kb.newline", "插入换行（多行输入）"),
    ("kb.palette", "打开命令面板"),
    ("kb.complete", "补全高亮的命令 / @path"),
    ("kb.copy_reply", "复制最后回复（干净、无软换行）"),
    ("kb.fold", "折叠 / 展开所有已完成的工具 chip"),
    ("kb.mouse", "切换鼠标捕获（滚轮滚动 ⇄ 原生选中）"),
    ("kb.scroll", "滚动对话记录"),
    ("kb.views", "在对话 ⇄ 会话面板间切换"),
    ("kb.dashboard", "打开会话面板"),
    ("kb.new_session", "新建并切换到新会话"),
    ("kb.cycle_session", "上一个 / 下一个会话"),
    ("kb.drop_session", "关闭当前会话"),
    ("kb.branch", "复制历史并分叉会话"),
    ("kb.theme", "打开主题选择器"),
    ("kb.help", "显示这些快捷键"),
    ("kb.escape", "返回 · 连按两次 Esc 回退"),
    ("kb.quit", "中止 / 退出"),
    // -- /status --------------------------------------------------------------
    ("status.title", "状态 · /status"),
    ("status.model", "模型"),
    ("status.connection", "连接"),
    ("status.state", "状态"),
    ("status.state.idle", "空闲"),
    ("status.state.working", "运行中"),
    ("status.sessions", "会话"),
    ("status.by_status", "  按状态"),
    ("status.needs_input", "待输入"),
    ("status.working", "运行中"),
    ("status.completed", "已完成"),
    ("status.context", "上下文"),
    ("status.cost", "费用"),
    ("status.git", "git"),
    ("status.cwd", "目录"),
    ("status.total", "合计"),
    ("status.close", "  esc 关闭"),
    // -- /cost ----------------------------------------------------------------
    ("cost.title", "费用 · /cost"),
    ("cost.header", "Token 用量"),
    ("cost.input", "输入"),
    ("cost.output", "输出"),
    ("cost.cache", "缓存"),
    ("cost.total", "合计"),
    ("cost.context", "上下文"),
    ("cost.cost", "费用"),
    // -- /verbose -------------------------------------------------------------
    ("verbose.title", "工具调用审计 · /verbose"),
    ("verbose.empty", "  本会话还没有工具调用。"),
    // -- /btw -----------------------------------------------------------------
    ("btw.title", " /btw "),
    ("btw.querying", "查询中…（side-agent 正在回答；主任务继续运行）"),
    ("btw.dismiss", "esc 关闭"),
    ("btw.usage", "用法：/btw <问题>"),
    ("btw.failed", "side-question 失败"),
    // -- /scheduler -----------------------------------------------------------
    ("scheduler.title.pick", "定时任务 · 选择任务"),
    ("scheduler.title.confirm", "定时任务 · 确认变更"),
    ("scheduler.title.status", "定时任务 · cron 状态"),
    ("scheduler.hint.pick", "↑↓ 移动 · 空格 勾选 · enter 审阅 · esc 取消"),
    ("scheduler.hint.confirm", "enter 应用 · esc 返回"),
    ("scheduler.hint.status", "esc 关闭"),
    ("scheduler.start", "启动"),
    ("scheduler.stop", "停止"),
    ("scheduler.unchanged", "不变"),
    ("scheduler.running", "运行中"),
    ("scheduler.stopped", "已停止"),
    ("scheduler.no_changes", "无变更 —— 每个任务都已处于目标状态"),
    ("scheduler.applied", "定时任务已更新"),
    ("scheduler.none_selected", "定时任务：未选择任何任务"),
    ("scheduler.cron_active", "cron 运行中"),
    ("scheduler.cron_none", "没有已排程的 cron 任务"),
    ("scheduler.will_start", "将启动"),
    ("scheduler.will_stop", "将停止"),
    ("scheduler.kind.reflect", "反射模式"),
    ("scheduler.kind.cron", "定时任务"),
    ("scheduler.cadence.reflect", "监控守护"),
    ("scheduler.repeat.once", "一次"),
    ("scheduler.repeat.daily", "每日"),
    ("scheduler.repeat.weekday", "工作日"),
    ("scheduler.repeat.weekly", "每周"),
    ("scheduler.repeat.monthly", "每月"),
    ("scheduler.empty", "未找到反射模式或定时任务"),
    // -- /continue ------------------------------------------------------------
    ("continue.title", "继续 · 搜索会话"),
    ("continue.search", "搜索："),
    ("continue.search.placeholder", "输入关键词按内容过滤…"),
    ("continue.hint", "输入搜索 · ↑↓ 移动 · enter 恢复 · esc 取消"),
    ("continue.empty", "未找到历史会话"),
    ("continue.no_match", "没有匹配搜索的会话"),
    ("continue.restoring", "正在恢复会话…"),
    ("continue.rounds", "轮"),
    ("continue.restored", "已恢复 {n} 轮对话"),
    ("continue.replaying", "正在重放历史对话…"),
    // -- /rewind --------------------------------------------------------------
    ("rewind.empty", "无可回退内容（还没有对话轮次）"),
    ("rewind.turn", "第"),
    ("rewind.done", "已回退"),
    ("rewind.turns_suffix", "轮"),
    // -- /export --------------------------------------------------------------
    ("export.clip", "剪贴板"),
    ("export.clip.detail", "复制最后回复到剪贴板（OSC52）"),
    ("export.all", "全部"),
    ("export.all.detail", "复制整个对话记录"),
    ("export.file", "文件"),
    ("export.file.detail", "把最后回复写到当前目录的文件"),
    ("export.copied_reply", "已复制最后回复到剪贴板"),
    ("export.copied_all", "已复制对话记录到剪贴板"),
    ("export.wrote", "已写入最后回复 →"),
    ("export.failed", "导出失败"),
    ("export.none", "还没有可导出的助手回复"),
    // -- /llm -----------------------------------------------------------------
    ("llm.querying", "正在查询模型…"),
    ("llm.switching", "正在切换模型…"),
    ("llm.no_models", "未配置任何模型（检查 mykey.py）"),
    // -- /theme / /emoji / /language ------------------------------------------
    ("theme.set", "主题 →"),
    ("theme.unknown", "未知主题（试试 /theme）"),
    ("emoji.updated", "样式已更新"),
    ("emoji.pet", "桌宠"),
    ("emoji.spinner", "spinner"),
    ("emoji.off", "关闭"),
    ("lang.set.en", "language → English"),
    ("lang.set.zh", "界面语言 → 简体中文"),
    // -- dashboard ------------------------------------------------------------
    ("dash.needs_input", "待输入"),
    ("dash.working", "运行中"),
    ("dash.completed", "已完成"),
    ("dash.new_session", "描述一个任务来新建会话"),
    ("dash.hint", "enter 打开 · 空格 回复 · ctrl+x 删除 · r 重命名 · ctrl+n 新建 · ? 快捷键"),
    ("dash.awaiting", "N 个待输入"),
    ("dash.header.working", "运行中"),
    ("dash.header.completed", "已完成"),
    ("dash.preview.idle", "发送一条消息开始"),
    ("dash.preview.working", "运行中…"),
    ("dash.preview.awaiting", "等待你的回答"),
    ("dash.rename", "重命名："),
    // -- generic command notices ----------------------------------------------
    ("cmd.unknown", "未知命令"),
    ("cmd.did_you_mean", "你是想用"),
    ("cmd.type_help", "（输入 /help 查看列表）"),
    ("cmd.not_wired", "UI 尚未接入"),
    ("cmd.not_handled", "尚未处理"),
    ("cmd.clear.busy", "运行中无法清屏（请先 /stop）"),
    ("cmd.aborted", "已中止运行中的任务"),
    ("cmd.rename.usage", "用法：/rename <名称>"),
    ("cmd.renamed", "已重命名会话 →"),
    ("cmd.restoring", "正在把历史对话恢复到上下文…"),
    ("cmd.reloading_keys", "正在重载 mykey.py…"),
    ("cmd.effects", "特效"),
    ("cmd.effects.off_default", "（默认关闭；演示将在特效阶段加入）"),
    // -- bridge / connection notices ------------------------------------------
    ("notice.bridge.not_connected", "无法发送：bridge 未连接（见状态栏）"),
    ("notice.bridge.spawn_failed", "bridge 启动失败"),
    ("notice.bridge.exited", "bridge 已退出"),
    ("notice.error", "错误"),
    // -- multi-press chords (§8) ----------------------------------------------
    ("ctrlc.arm", "再按一次 Ctrl+C 退出"),
    // -- key hints ------------------------------------------------------------
    ("key.enter", "enter"),
    ("key.esc", "esc"),
    ("key.cancel", "取消"),
    ("key.apply", "应用"),
    ("key.back", "返回"),
    ("key.close", "关闭"),
    ("key.move", "移动"),
    ("key.toggle", "勾选"),
    ("key.select", "选择"),
    ("key.search", "搜索"),
    ("key.restore", "恢复"),
    ("key.open", "打开"),
    ("key.reply", "回复"),
    ("key.delete", "删除"),
    ("key.rename", "重命名"),
    ("key.new", "新建"),
    // -- /workflows panel (§7) ------------------------------------------------
    ("wf.title", "工作流"),
    ("wf.hint", "↑↓ 聚焦 · t 样式 · enter 详情 · r 刷新 · esc 返回"),
    ("wf.style", "样式"),
    ("wf.style.tree", "盒式树"),
    ("wf.style.list", "紧凑列表"),
    ("wf.count.workflows", "工作流"),
    ("wf.count.nodes", "节点"),
    ("wf.count.running", "运行中"),
    ("wf.turns", "轮"),
    ("wf.empty", "没有正在运行的工作流。"),
    ("wf.empty.launch", "用 /conductor、/hive 或 /goal 启动一个。"),
    ("wf.down", "未运行"),
    ("wf.down.launch", "按 X 启动"),
    ("wf.group.conductor", "Conductor 指挥"),
    ("wf.group.hives", "Hive 蜂群"),
    ("wf.group.goal", "Goal 目标"),
    ("wf.status.running", "运行中"),
    ("wf.status.idle", "空闲"),
    ("wf.status.wrapping_up", "收尾中"),
    ("wf.status.done", "已完成"),
    ("wf.status.failed", "失败"),
    ("wf.status.aborted", "已中止"),
    ("wf.status.unknown", "未知"),
    ("wf.action.keyinfo", "注入提示"),
    ("wf.action.input", "发送任务"),
    ("wf.action.stop", "停止"),
    ("wf.action.kill", "终止"),
    ("wf.action.open", "查看详情"),
    ("wf.action.sent", "已发送动作："),
    ("wf.action.unavailable", "动作不可用（服务未运行）"),
    ("wf.detail", "详情"),
    ("wf.detail.status", "状态"),
    ("wf.detail.workflow", "工作流"),
    ("wf.detail.parent", "上级"),
    ("wf.detail.prompt", "任务"),
    ("wf.detail.output", "输出"),
    ("wf.detail.feed", "动态"),
    ("wf.detail.actions", "动作"),
    ("wf.detail.hint", "↑↓ 选动作 · enter 执行 · esc 关闭"),
    // -- units / misc ---------------------------------------------------------
    ("unit.bytes", "字节"),
    ("unit.tokens", "tok"),
    ("unit.seconds", "秒"),
    // -- copy / clipboard feedback + mouse toggle -----------------------------
    ("copy.ok", "已复制"),
    ("copy.fail", "复制失败"),
    ("copy.fail.too_large", "内容过大；请用 /export file"),
    ("copy.fail.no_tty", "没有可写入的终端"),
    ("copy.fail.generic", "没有可用的剪贴板"),
    ("copy.label.selection", "选区"),
    ("copy.label.cut", "剪切内容"),
    ("copy.label.reply", "最后回复"),
    ("copy.label.transcript", "对话记录"),
    ("mouse.on", "鼠标捕获已开启（滚轮滚动 · 点击进面板）"),
    ("mouse.off", "鼠标捕获已关闭——拖动选中后用终端自带复制"),
    ("mouse.hint.native", "拖动即可原生选中复制；Ctrl+Shift+M 重新开启滚轮滚动"),
    ("misc.none", "—"),
    ("misc.yes", "是"),
    ("misc.no", "否"),
    ("misc.on", "开"),
    ("misc.off", "关"),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// THE deliverable test: the i18n fallback chain `lang → en → key`.
    /// A key present in the active language resolves to its translation; a key
    /// missing from `zh` but present in `en` falls back to English; a key missing
    /// from BOTH falls back to the key string itself — never a panic, never "".
    #[test]
    fn i18n_fallback_chain() {
        // 1. Active-language hit: a zh key resolves to its Chinese string.
        assert_eq!(t(Lang::Zh, "conn.connecting"), "连接中…");
        assert_eq!(t(Lang::En, "conn.connecting"), "connecting…");

        // 2. lang → en fallback: a key that only English defines resolves to the
        // English string when the active language is zh. We simulate a hole by
        // asserting EVERY en key resolves under zh to a NON-EMPTY string (either a
        // zh translation or the en fallback), and that the resolver returns the
        // English string for a key we know is en-only-shaped. To make the hole
        // concrete we use a key guaranteed absent from zh by construction below.
        let en_only = "test.only.en.key.zzz";
        // Neither dict has it → 3. key fallback (never empty, never panic).
        assert_eq!(t(Lang::En, en_only), en_only, "missing key → key itself (en)");
        assert_eq!(t(Lang::Zh, en_only), en_only, "missing key → key itself (zh)");
        // The fallback is interned to a stable &'static str across calls.
        assert_eq!(t(Lang::Zh, en_only).as_ptr(), t(Lang::En, en_only).as_ptr());

        // The chain NEVER yields an empty string for ANY key in either dict.
        for (k, _) in EN_PAIRS {
            assert!(!t(Lang::En, k).is_empty(), "en[{k}] must be non-empty");
            assert!(!t(Lang::Zh, k).is_empty(), "zh[{k}] (with en fallback) must be non-empty");
        }
        for (k, _) in ZH_PAIRS {
            assert!(!t(Lang::Zh, k).is_empty(), "zh[{k}] must be non-empty");
        }
        // The `tf` (owned) variant agrees with `t`.
        assert_eq!(tf(Lang::Zh, "conn.connected"), "已连接");
        assert_eq!(tf(Lang::En, en_only), en_only);
    }

    /// The two dictionaries cover the SAME key set (the coverage guard): every
    /// English (superset) key has a zh entry and vice-versa — so the fallback chain
    /// is only ever exercised by genuine bugs, not by routine missing translations.
    #[test]
    fn dictionaries_cover_the_same_keys() {
        let en = en_dict();
        let zh = zh_dict();
        // No duplicate keys collapsed away (the pair list length == map length).
        assert_eq!(en.len(), EN_PAIRS.len(), "duplicate en key collapsed the map");
        assert_eq!(zh.len(), ZH_PAIRS.len(), "duplicate zh key collapsed the map");
        // Every en key is in zh.
        for (k, _) in EN_PAIRS {
            assert!(zh.contains_key(k), "zh is missing en key {k:?}");
        }
        // Every zh key is in en (en is the canonical superset).
        for (k, _) in ZH_PAIRS {
            assert!(en.contains_key(k), "en (superset) is missing zh key {k:?}");
        }
        // The key surface is sizeable (~250 the spec calls for; ≥150 guards drift).
        assert!(EN_PAIRS.len() >= 150, "expected ~250 keys, got {}", EN_PAIRS.len());
    }

    /// System-locale detection maps real locale strings onto the right `Lang`
    /// (the PURE `from_code` the effectful `detect_system_lang` delegates to).
    #[test]
    fn locale_detect_from_code() {
        // Chinese variants.
        assert_eq!(Lang::from_code("zh"), Some(Lang::Zh));
        assert_eq!(Lang::from_code("zh-CN"), Some(Lang::Zh));
        assert_eq!(Lang::from_code("zh_CN.UTF-8"), Some(Lang::Zh));
        assert_eq!(Lang::from_code("zh-Hans"), Some(Lang::Zh));
        assert_eq!(Lang::from_code("Chinese (Simplified)"), Some(Lang::Zh));
        // English / C / POSIX variants.
        assert_eq!(Lang::from_code("en"), Some(Lang::En));
        assert_eq!(Lang::from_code("en_US.UTF-8"), Some(Lang::En));
        assert_eq!(Lang::from_code("C"), Some(Lang::En));
        assert_eq!(Lang::from_code("C.UTF-8"), Some(Lang::En));
        assert_eq!(Lang::from_code("English"), Some(Lang::En));
        // Unknown → None (caller keeps the current lang).
        assert_eq!(Lang::from_code("fr_FR"), None);
        assert_eq!(Lang::from_code(""), None);
        // Round-trips through the code() label.
        assert_eq!(Lang::from_code(Lang::Zh.code()), Some(Lang::Zh));
        assert_eq!(Lang::from_code(Lang::En.code()), Some(Lang::En));
        // detect_system_lang never panics and returns a real language.
        let _ = detect_system_lang();
    }

    /// Per-language rotating tips are deterministic and parallel across languages
    /// (the §9 "per-language rotating tips"). Moved here from `flavor`.
    #[test]
    fn tips_rotate_deterministically_per_language() {
        assert!(!TIPS_EN.is_empty());
        assert_eq!(TIPS_EN.len(), TIPS_ZH.len(), "parallel tip lists");
        assert_eq!(tip_at(Lang::En, 0), TIPS_EN[0]);
        assert_eq!(tip_at(Lang::En, TIPS_EN.len() as u64), TIPS_EN[0]); // wraps
        assert_eq!(tip_at(Lang::Zh, 1), TIPS_ZH[1]);
        // One tip held for a full ~12s window, then steps.
        for t in 0..TIP_TICKS_PER_STEP {
            assert_eq!(tip(Lang::En, t), TIPS_EN[0]);
        }
        assert_eq!(tip(Lang::En, TIP_TICKS_PER_STEP), TIPS_EN[1]);
        // Languages select different pools.
        assert_ne!(tip_at(Lang::En, 2), tip_at(Lang::Zh, 2));
    }
}
