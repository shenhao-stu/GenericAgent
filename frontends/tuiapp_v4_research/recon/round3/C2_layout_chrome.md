# C2 — Splash / header / composer / footer relayout · input prefix · spinner+pet+tab

Audit target: `D:/GenericAgent/frontends/tuiapp_v4` (Rust + ratatui). All cites are file:line in that tree unless prefixed `temp/claude-code` (CC reference) or a Codex URL. RESEARCH+AUDIT only.

Covers Q1 (history-input full-row `rgb(58,58,58)` + `❯` prefix), Q7 (splash slogan + header + above-composer completion line + 2-row below-composer + `/keybindings`), Q9 (spinner=`⠿`, emoji→pet-only, tab=bear).

---

## Findings (file:line bugs, root cause not symptom)

**F1 — Q1: user history is NOT a full-row `rgb(58,58,58)` band; it's a `❯ `-less narrow band. The `❯` prefix the user asks for is missing.**
`components/mod.rs:357-373` `user_band_line()` builds the band as `lead=" "` (one space) + clipped body + right-pad — the band IS full-width `Token::UserBand` (verified by test `user_row_has_band_bg` :1078), but the lead is a bare space, not `❯ `. The test at :1121 actively *asserts* `!row.trim_start().starts_with('❯')`. So Q1 ("整条 rgb(58,58,58) 并包含 ❯ 前缀") is half-done: bg is right, prefix is wrong. Root cause: §2.1 of the old `redesign_cc.md` deliberately replaced the `❯ hello` gutter with a CC-style bare band; Q1 reverses that decision — it wants the band AND a leading `❯`.
Note collision: `gutter_for(BlockRole::User)` at :378 still returns `("❯ ", …)` but that branch is dead for users — :312-315 routes `BlockRole::User` to `user_band_line` before the gutter code at :319 runs. So the `❯` constant exists but is never drawn.

**F2 — Q7: header shows the literal `◆ GenericAgent · tui_v4`, not a designed slogan, and omits llm-channel / session.**
`components/mod.rs:167-191` `render_header` renders exactly `◆ GenericAgent · tui_v4   ·   <model>   ·   <cwd>`. The `>_ GenericAgent` the user wants replaced lives in **v3** (`frontends/tui_v3.py:3682`), not v4 — v4 already moved past `>_` but landed on a plain product-name string, not a slogan, and shows neither the LLM channel (simplified) nor `session[name]`. `i18n` key `app.name` (`i18n/mod.rs:275`) is also `"GenericAgent · tui_v4"`. So Q7's header spec (slogan + llm渠道[简洁版] + model + directory + session) is unmet.

**F3 — Q7: there is NO above-composer completion line.** Grep for the v2 `"⠿ Patched for 1m 46s · ↑ 47.3k · ↓ 472"` pattern in v4 → zero hits. The spinner band (`render_spinner` :387-438) only renders **while `app.busy`** (gated at :82 `show_spinner = app.busy`, constraint pushed only then at :91-93). On completion the band vanishes — there is no settled "ran for Xs · ↑in ↓out" summary. v2's equivalent is the topbar dot freeze (`tuiapp_v2.py:2248-2256` `render_topbar`, `running {_fmt_elapsed}` / `done`), and the gerund+token readout is the spinner; the user wants that frozen line shown **above the input box** after each turn. Required state already exists: `app.tokens`/`tok_in`/`tok_out`/`turn_started_ms` (`app/mod.rs:407-413,392`) — but `turn_elapsed_ms` returns 0 when `!busy` (:597-603), so a "last turn duration" needs a new frozen field.

**F4 — Q7: the below-composer area is ONE row (hints), not the required TWO rows (row1 session-info, row2 `⎿ Tips`).**
`components/mod.rs:99` pushes a single `Constraint::Length(1)` for hints; `render_hints` (:196-246) crams keybinding pairs (left) + rotating tip (right) onto that one row. Q7 wants: row1 = runtime session info `llm, model, effort, ctx, branch`; row2 = `⎿ Tips` dynamic. Currently the session info that should be row1 is instead in the **footer** (`render_footer` :591-671, right side: model · ctx · cost · git) and the tip is right-aligned on the hint row, not a dedicated `⎿ Tips` row. The `⎿`/`└ Tip:` shape exists in v2 (`tuiapp_v2.py:172-181` `_tip_line`, prefix `"└ "`) and v3 — v4 dropped the leader glyph.

**F5 — Q7: the ugly `❯ chat` is produced in the footer mode indicator.**
`components/mod.rs:598-607` `render_footer`: left side = `("❯ ", Token::Dim)` mark + `mode_word`. `mode_indicator` (:676-684) returns `footer.mode.chat` → `"chat"` (`i18n/mod.rs:281`) in the idle case. So the literal `❯ chat` the user calls out is drawn here. (The comment at :595-597 claims they already removed the *background* pill, but the `❯ chat` text remains.) Q7: "不要再出现 ❯ chat".

**F6 — Q9: emoji can appear in the spinner band via PetStyle, and the spinner default is the arc set `◜◠◝◞◡◟`, not `⠿`.**
- Spinner default = `SpinnerStyle::Arc` (`flavor/mod.rs:26-27,35`), frames `◜◠◝◞◡◟`. Q9 wants the spinner to be *only* `⠿`. Note `⠿` (U+283F, braille all-dots) is NOT in any current set: `ARC_FRAMES` :35, `BRAILLE_FRAMES` :37 (`⠋⠙⠹…`), `PULSE_GLYPHS` :39. So a static `⠿` glyph must be introduced.
- The pet (kaomoji, not emoji by default — `PetStyle::Off` default :233) is rendered *inside the spinner band* at `components/mod.rs:403-408`. The faces are kaomoji (`ʕ•ᴥ•ʔ` etc., :238-275), not emoji — so "emoji 只出现在 pet" is mostly satisfied already EXCEPT the user's intent is that the spinner row carry NO decorative glyph other than `⠿`, while the *pet* is the only place a face/emoji shows. Current code couples pet+spinner on the same row, which is fine, but the spinner glyph itself is the arc, not `⠿`. There is no actual emoji (😀-class) anywhere — kaomoji only. If Q9 literally means "emoji", the pet pools would need an emoji variant; given memory `redesign_cc.md §2.6` ("NOT emoji pet by default — kaomoji OK"), interpret Q9 as: spinner glyph fixed to `⠿`, decorative face stays the pet.

**F7 — Q9: terminal tab title does NOT default to bear; it uses the spinner glyph + product name.**
`app/mod.rs:662-672` `terminal_title()` → busy: `"{glyph} GenericAgent · {model}"` (glyph = arc spinner char); idle: `"GenericAgent · {model}"`. No bear. Q9: "终端 tab 标题默认用 bear" — the OSC0 title (and/or OSC-21337 tab label) should lead with the bear kaomoji `ʕ•ᴥ•ʔ`. `util/osc.rs:58-64` `build_title` strips control chars only (kaomoji survives — test :105 proves CJK passes), so `ʕ•ᴥ•ʔ` is safe to emit.

---

## Competitor patterns (CC / Codex / v2 / v3, with file cites)

**Splash (v3, the thing being replaced):** `frontends/tui_v3.py:3671-3696` `_make_banner_lines` — a rounded box (`╭╮╰╯`) with rows: `>_ GenericAgent` / blank / `model  <name>  <llm_hint>` / `directory  <cwd>` / `session  <sess>`, then a blank + `  Tip: …`. This is the literal structure Q7 says to keep (model/directory/session) but with a better slogan than `>_`.

**Splash (CC):** `temp/claude-code/src/components/LogoV2/WelcomeV2.tsx` — `WELCOME_V2_WIDTH=58` box, `<Text color="claude">Welcome to Claude Code</Text> v<VERSION>`, then a multi-line ░▒▓█ ASCII claw mascot. Left-aligned, fixed width. Takeaway for Q7 "整体左对齐": one accent title line + dim version, left-aligned, then info rows — no center alignment.

**Above-composer completion line (v2 — the exact reference Q7 quotes):** the gerund spinner readout. `tuiapp_v2.py:2198-2208` `_gerund_color` (tiered cool→red ramp by elapsed+tokens); the running indicator with `↑/↓` token deltas and elapsed is the spinner. On done the per-session dot freezes to `done` (`render_topbar` :2252-2254). The phrase "Patched for 1m 46s · ↑ 47.3k · ↓ 472" = `<Gerund> for <elapsed> · ↑<in> · ↓<out>` — gerund pool is the same one v4 already ships (`flavor/mod.rs:154-189`, `Patched` is at :180).

**Spinner status group (CC):** `temp/claude-code/src/components/Spinner/SpinnerAnimationRow.tsx:202-225` — parts wrapped in dim `(` … `)`: `spinnerSuffix · timer · ↑/↓ tokens · thinking`. Tokens use `figures.arrowDown`/`arrowUp` (↑/↓) (:168, `SpinnerModeGlyph` :232-263). Spinner glyph (`SpinnerGlyph.tsx:7,49`) = `getDefaultCharacters()` braille set forward+reversed — **never an emoji**, exactly Q9's principle. v4's `render_spinner` already mirrors this `(…)` group (:417-435) — good; it just needs the glyph swapped to `⠿` and the same readout frozen above the composer on done.

**Footer / mode indicator (CC):** `temp/claude-code/src/components/PromptInput/PromptInputFooterLeftSide.tsx:317-318` — bash mode renders `<Text color="bashBorder">! for bash mode</Text>`; mode is plain colored text + ` · ` separators (:472-477 `<Text dimColor> · </Text>`), and the idle fallback is `? for shortcuts` (:409-412) — **never `❯ chat`**. CC's whole left footer is hint-or-mode text, no `❯` caret on a "chat" word. This is the model for F5: drop the `❯ chat`, show either nothing or a real mode word only when non-default.

**Codex composer/footer (peer, file cites):** `recon/codex_tui_patterns.md` §6 + §4:
- Status indicator (one row above composer, busy only): `codex-rs/tui/src/status_indicator_widget.rs` — `spinner · "Working" shimmer · (elapsed • esc to interrupt)`. `fmt_elapsed_compact` (`:65`): `0s` / `1m 00s` / `1h 00m 00s`. Pausable work-time timer (`:160-199`). **Spinner is a single `•` shimmer, not emoji** (`motion.rs`).
- Footer hints centralized in `codex-rs/tui/src/bottom_pane/footer.rs` (~2k lines); two-line vs single-line vs hint-only fallback by width (per WebSearch deepwiki "Status Line and Footer Rendering"); keys via `key_hint::KeyBinding`. Stable-height rule: never add/remove footer rows dynamically (PR #19901) — reuse the existing row. → directly supports F4's "2 fixed rows" being safer than the current conditional single row.
- `/statusline` (PR #10546) makes the status-line items user-ordered; default = `model-with-reasoning` + `current-dir`. → Q7 row1 (`llm, model, effort, ctx, branch`) is the GA analogue.

**v2 tip leader glyph:** `tuiapp_v2.py:172-181` `_tip_line` → `"└ "` + `"Tip: "` + body, dim `#6e7681`. Q7 wants `⎿ Tips` (the `⎿` is the rounded variant of `└`). v4's tip currently has no leader (`render_hints` :238-244 appends bare tip text).

---

## Fix design (Rust sketches: the actual changed lines / new fn signatures)

### Fix A (Q1) — `❯ `-prefixed full-width `rgb(58,58,58)` history input
`components/mod.rs` `user_band_line` (:357-373). Swap the bare-space lead for a styled `❯ ` and shrink the body budget by 2. Whole row stays `Token::UserBand` bg.

```rust
fn user_band_line<'a>(text: &str, width: u16, theme: &Theme) -> Line<'a> {
    let band = Style::default().bg(theme.color(Token::UserBand)).fg(theme.color(Token::Text));
    let w = width as usize;
    let prompt = "❯ ";                                  // Q1: visible prefix inside the band
    let body = clip_to(text, w.saturating_sub(2));       // 2 cells for "❯ "
    let used = 2 + unicode_width::UnicodeWidthStr::width(body.as_str());
    let pad = w.saturating_sub(used);
    Line::from(vec![
        Span::styled(prompt.to_string(), band),          // same bg → still one solid band
        Span::styled(body, band),
        Span::styled(" ".repeat(pad), band),
    ])
}
```
Test update: `user_band_line_spans_width_with_band_bg` (:1051) — change `joined.starts_with(" hello")` → `starts_with("❯ hello")`. `user_row_has_band_bg` (:1078) — DELETE the `:1121` assertion `!row.trim_start().starts_with('❯')` (it now contradicts the spec); keep the every-cell-bg assertion. Decision note for impl: Q1 wants `❯` charcoal-on-band; if a brighter caret is wanted, give the prompt span `.fg(Token::Claude)` — but the whole-row bg assertion must stay green, so keep bg = `UserBand`. The query says "整条颜色渲染 rgb(58,58,58)" → keep fg=Text for visual uniformity.

### Fix B (Q7) — header: slogan + llm[简洁] · model · directory · session
`components/mod.rs` `render_header` (:167-191). Replace the product-name spans with a slogan + the four info fields. Add an `llm_channel` accessor (the simplified channel name) — derive from `app.model` via a new pure helper or reuse `truncate_model`. Session name: read `app.sessions.active`'s name (`app/session.rs`). New signature unchanged; body:

```rust
let slogan = SLOGAN;                                  // pick from the 5 options below (const &str)
let llm = llm_channel(app.model.as_deref());          // NEW pure fn: "OhMyAPI" / "ModelScope" / "—"
let model = truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP);
let cwd = compact_cwd(&app.cwd, 30);
let sess = app.sessions.active_name();                // NEW on SessionMap → &str, falls back to "main"
let dim = Style::default().fg(theme.color(Token::Dim));
let key = |s: &'static str| Span::styled(s, dim);
let val = |s: String| Span::styled(s, Style::default().fg(theme.color(Token::Text)));
let spans = vec![
    Span::styled(slogan, Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD)),
    key("   llm "),  Span::styled(llm, Style::default().fg(theme.color(Token::Suggestion))),
    key("   model "), Span::styled(model, Style::default().fg(theme.color(Token::Claude))),
    key("   dir "),  val(cwd),
    key("   session "), Span::styled(sess.to_string(), Style::default().fg(theme.color(Token::Suggestion))),
];
frame.render_widget(Paragraph::new(Line::from(spans)), area);   // left-aligned (Q7)
```
`SLOGAN` const — 3–5 monochrome options (single accent color via `Token::Claude`, no emoji, terse, left-aligned). Pick ONE:
1. `▲ GenericAgent` — a single filled triangle "play/forward" mark; minimal, mono, distinct from CC's `◆`/`✻` and v3's `>_`.
2. `◇ GenericAgent · your terminal, with an agent` — diamond mark + one-line tagline.
3. `❯❯ GenericAgent` — double-caret = "fast-forward / shell-native"; echoes the `❯` input prefix for cohesion.
4. `GA ▸ generic agent, native terminal` — initials + a small right-triangle, lowercase tagline.
5. `∴ GenericAgent` — "therefore" glyph = reasoning; one mono char, very clean.
(Recommendation: option 1 or 3 — one glyph + name, fits one line beside the four fields; option 3 reuses `❯` so the header rhymes with the composer+history prefix.)

### Fix C (Q7) — above-composer completion line (settled "ran for Xs · ↑in ↓out")
Two parts: (1) freeze the turn duration on `MessageEnd`; (2) render a 1-row band above the composer when not busy and a turn has run.

`app/mod.rs`: add `pub last_turn_ms: Option<u64>` to `AppState` (near :392). In `apply_frame` `MessageEnd` (:764-793), before `self.busy = false`, stamp it:
```rust
self.last_turn_ms = Some(now_ms.saturating_sub(self.turn_started_ms));
```
`components/mod.rs` `render_cockpit` — add a constraint above the composer mirroring the spinner band, shown when `!show_spinner && app.last_turn_ms.is_some()`:
```rust
let show_done = !show_spinner && app.last_turn_ms.is_some();
if show_done { constraints.push(Constraint::Length(1)); }   // place right before composer
```
New `render_done_line(frame, area, app, theme)` (mirror `render_spinner`'s `(…)` group, dim chrome + bright numbers, gerund frozen, no animation):
```rust
fn render_done_line(frame, area, app, theme) {
    let secs = app.last_turn_ms.unwrap_or(0) as f64 / 1000.0;
    let dim = Style::default().fg(theme.color(Token::Dim));
    let text = Style::default().fg(theme.color(Token::Text));
    let mut spans = vec![
        Span::styled("⠿ ", Style::default().fg(theme.color(Token::Success))),   // Q9: ⠿ only, mint=done
        Span::styled(format!("{} ", flavor::gerund(app.turn_index)), text),     // a frozen gerund (e.g. "Patched")
        Span::styled(format!("for {}", fmt_dur(secs)), dim),
    ];
    if app.tok_in.is_some() || app.tok_out.is_some() {
        spans.push(Span::styled(" · ↑ ".into(), dim));
        spans.push(Span::styled(human_count(app.tok_in.unwrap_or(0)), text));
        spans.push(Span::styled(" · ↓ ".into(), dim));
        spans.push(Span::styled(human_count(app.tok_out.unwrap_or(0)), text));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}
```
`fmt_dur` = a new pure helper matching v2's `_fmt_elapsed` / Codex `fmt_elapsed_compact`: `1m 46s` form. (`human_count` already exists :902, gives `47.3k`.) For the frozen gerund pick a turn-stable index (e.g. a per-turn counter) so it doesn't jitter; or hardcode "Done" if a verb feels off. Clear `last_turn_ms` on the next `MessageBegin` (so it shows only between turns).

### Fix D (Q7) — two below-composer rows (row1 session-info, row2 `⎿ Tips`)
`components/mod.rs` `render_cockpit` (:99): replace the single hint constraint with two.
```rust
constraints.push(Constraint::Length(1)); // row1: runtime session info
constraints.push(Constraint::Length(1)); // row2: ⎿ Tips
```
Index both (`let info = chunks[i]; i+=1; let tips = chunks[i];`) and call two new fns. **Move** the session-info fields OUT of the footer (Fix E) into row1; keep keybinding hints either folded into row1 or dropped (Q7 routes shortcuts to `/keybindings`).

```rust
/// ROW 1: runtime session info — llm · model · effort · ctx · branch (left-aligned, dim chrome).
fn render_session_info(frame, area, app, theme) {
    let dim = Style::default().fg(theme.color(Token::Dim));
    let sep = Span::styled("  ·  ", dim);
    let llm = llm_channel(app.model.as_deref());
    let model = truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP);
    let effort = app.effort_label().unwrap_or("—");
    let ctx = app.context_percent.map(|p| format!("ctx {p:.0}%")).unwrap_or_else(|| "ctx —".into());
    let branch = app.git_branch.as_deref().unwrap_or("—");
    let spans = vec![
        Span::styled(llm,   Style::default().fg(theme.color(Token::Suggestion))), sep.clone(),
        Span::styled(model, Style::default().fg(theme.color(Token::Claude))),     sep.clone(),
        Span::styled(effort.to_string(), Style::default().fg(theme.color(Token::PlanMode))), sep.clone(),
        Span::styled(ctx,   dim),                                                 sep.clone(),
        Span::styled(branch.to_string(), Style::default().fg(theme.color(Token::Suggestion))),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// ROW 2: `⎿ Tips` — dynamic rotating tip (deterministic by tick), leader glyph like v2/v3.
fn render_tips(frame, area, app, theme, now_ms) {
    let dim = Style::default().fg(theme.color(Token::Dim));
    let tip = flavor::tip(app.lang, now_ms / 100);                 // already deterministic
    let body = clip_to(tip, (area.width as usize).saturating_sub(2));
    let spans = vec![Span::styled("⎿ ", dim), Span::styled(body, dim)];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}
```
The existing armed-`Ctrl+C` hint (`render_hints` :228-233) should move onto row2 as a transient override of the tip (same as today), in `Token::Warning`. Keybinding pairs (:203-211) are removed from the chrome and surfaced via `/keybindings` (Fix F).

### Fix E (Q7/F5) — footer: kill `❯ chat`
Two options; **prefer demoting the footer entirely** since Fix D row1 now owns the session info:
- Minimal (kill the eyesore only): `components/mod.rs:598-607` — drop the `❯ chat` for the idle/chat case. Only render a mode word when NOT in the default chat state (CC's model — `mode_indicator` returns a word only for bash/running):
```rust
let left: Vec<Span> = match () {
    _ if app.composer.is_shell_mode() => vec![Span::styled("! ", Style::default().fg(theme.color(Token::ShellAccent))),
                                              Span::styled(t(app.lang,"footer.mode.bash"), Style::default().fg(theme.color(Token::ShellAccent)))],
    _ if app.busy => vec![Span::styled(t(app.lang,"footer.mode.running"), Style::default().fg(theme.color(Token::Warning)))],
    _ => Vec::new(),                                  // idle: NOTHING (no `❯ chat`)
};
```
- Cleaner (recommended with Fix D): remove the footer row constraint at :98 altogether and fold the connection chip (the only N1-critical bit, :610-621) into row1's tail. Net effect: the model/ctx/cost/git that footer drew (:627-668) is now row1; the `❯ chat` disappears with the footer.
Whichever path: the `mode_indicator` chat branch (:682) and the `❯ ` mark (:601) are the lines to delete.

### Fix F (Q7) — bind shortcuts help to `/keybindings` or `ctrl+/`
The keybinding pairs currently inlined in `render_hints` (:203-211) move into a help overlay. `app/mod.rs` already has an `Overlay::Help` (:48) and i18n `help.*` keys (`i18n/mod.rs:322-327`). Add `Overlay::Keybindings` (or reuse Help) and register `/keybindings` in `commands/registry.rs`; map `ctrl+/` in the key handler to open it. The overlay renders the `pairs` table from :203-211 plus the magic-prefix line (`help.magic` :326). No new render machinery — it's an overlay like the others (`overlay::render`, :69).

### Fix G (Q9) — spinner glyph = `⠿`, decorative face = pet only
`flavor/mod.rs`: the simplest, lowest-blast-radius change is a static glyph. Either (a) add a `Static` variant, or (b) since Q9 says "spinner 就只用 ⠿", make the spinner a CONSTANT in the renderer and stop indexing frames:
- `components/mod.rs:391` change `let glyph = app.spinner_style.glyph(tick);` → `let glyph = '⠿';` (drop the per-tick frame index). This makes the busy spinner a steady `⠿`; the *animation* then comes from the pet blink (:403-408) + the gerund/heat ramp, which is what the user wants ("spinner 就只用 ⠿").
- Keep pet rendering at :403-408 (kaomoji = the only face). No emoji exists today; if literal emoji is required, add an emoji pet pool in `flavor/mod.rs` parallel to `PETS_BEAR` — but per memory `redesign_cc.md §2.6`, kaomoji-as-pet is the intended reading; leave `PetStyle::Off` default and let `/emoji` opt into `bear`.
- If you want to keep `SpinnerStyle` selectable, add `'⠿'` as the `Arc`-replacement default frame: `ARC_FRAMES` (:35) → a 1-element `&['⠿']`, but that defeats the picker. Cleaner: leave the sets, hardcode `⠿` at the call site (above). Update test `arc_is_the_default_and_not_the_cc_asterisk` (:367) is unaffected; add a test asserting `render_spinner` emits `⠿` (TestBackend scan like :1078).

### Fix H (Q9) — terminal tab default = bear
`app/mod.rs:662-672` `terminal_title()`. Lead with the bear kaomoji instead of the spinner glyph:
```rust
pub fn terminal_title(&self) -> String {
    let model = self.model.as_deref().unwrap_or("");
    let bear = crate::flavor::PETS_BEAR[0][0];          // "ʕ•ᴥ•ʔ" — Q9: tab defaults to bear
    if self.busy { format!("{bear} GenericAgent · {model}") }
    else         { format!("{bear} GenericAgent · {model}") }   // bear in both states; or drop model when idle
}
```
(`PETS_BEAR` is `pub` already, :238. `build_title` strips only control chars, :59-62, so the kaomoji passes — test :105 confirms CJK survives.) Update test `tab_status_and_title_track_state` (:1628): the `:1632` assertion `terminal_title().starts_with("GenericAgent")` must become `.contains("GenericAgent")` and the busy-glyph assertion at :1640 (`!starts_with("GenericAgent")`) still holds because the bear leads. Optionally vary the bear by heat tier while busy (reuse `flavor::pet`), but a static bear satisfies "默认用 bear".

---

## Review-principle violations (cite principle # + file:line)

- **P10 (视觉均匀) + P4 (变化半径):** `render_hints` (`components/mod.rs:196-246`) packs 7 keybinding pairs + a right-aligned rotating tip + an armed-Ctrl+C override into ONE row with manual width math (:220-244). It's a jagged, do-everything function; Q7 splits it into row1/row2. The manual `avail`/`pad` arithmetic (:234-242) is the kind of ad-hoc layout Codex centralizes in `footer.rs`. Refactor per Fix D.
- **P6 (约束写进代码) + P4:** the user-band prefix decision is encoded as a *negative test assertion* (`components/mod.rs:1121` `assert!(!row…starts_with('❯'))`) rather than a positive invariant. When Q1 flips the requirement, that test silently enforces the wrong thing. A constraint test should assert what the band IS (`starts_with("❯ ")` + full-row bg), not police a glyph out.
- **P8 (一致且不意外):** `❯` is used for three different meanings — composer prompt (:466 `mark="❯ "`), footer mode caret (:601), palette/file-picker cursor (:748,785). Meanwhile the user-row gutter constant `("❯ ",…)` at :378 is dead (F1). After Fixes A/E the `❯` family should mean exactly "input/prompt" (composer + history band), and the footer's `❯ chat` (a *status*, not a prompt) is removed — restoring one meaning per glyph.
- **P12 (功能越多代码越短) / P2 (局部可推理):** session info is currently duplicated across header (`render_header` model/cwd :170-188) and footer (`render_footer` model/ctx/cost/git :627-668). Fix B+D+E consolidate the live session fields into header + row1 and delete the footer, so adding a field touches one place, not two. Right now changing the model label means editing both `truncate_model` call sites.
- **P9 (注释极简):** `render_footer` carries a 4-line justification comment (:595-597) for *why* the inverse pill was removed — but the `❯ chat` text it left behind contradicts the comment's intent. Either the code or the comment is stale (the code). Fix E resolves both.

---

## Open questions / risks

1. **Q9 "emoji"**: GA's "pet" is kaomoji (`ʕ•ᴥ•ʔ`), not Unicode emoji. Memory `redesign_cc.md §2.6` explicitly says kaomoji pet is OK, emoji pet is not. I read Q9 as "spinner glyph = `⠿` only; the decorative *face* is confined to the pet (kaomoji)". If the user literally means emoji (😀), that's a NEW pet pool + a reversal of the documented default — confirm before adding.
2. **`⠿` width/rendering**: `⠿` (U+283F) is single-width braille; renders on every modern terminal but may show as tofu on ancient cp437 Windows consoles. Low risk (v3/v4 already ship braille `⠋⠙…`). The completion line (Fix C) also uses `⠿` in mint to signal "done".
3. **Above-composer line placement vs spinner band**: both occupy the same logical slot (just-above-composer). They are mutually exclusive (`busy` xor `done`), so one shared constraint with a branch is cleaner than two — but keep them as two `if` pushes for readability (P15). Risk: if `last_turn_ms` isn't cleared on the next `MessageBegin`, the done-line lingers a frame into the new turn; clear it at :735.
4. **Session name source**: row1/header need the active session's display name. `app.sessions` is a `SessionMap` (`app/session.rs`); confirm it exposes an `active_name()` / the active `Session.name`. If not, add a 1-line accessor (read-only). Not in this audit's file set — flag for impl.
5. **`llm_channel` simplification**: Q7 wants "llm 渠道[简洁版]". The model string is a MixinSession pipe-list; `truncate_model` (:841) gives the primary *model*, not the *channel/provider*. Deriving "OhMyAPI"/"ModelScope" needs a provider map (likely from `mykey.py`/bridge `Ready` frame) that the bridge doesn't currently send. Risk: there may be no channel field available — fallback to the primary model name, or extend `CoreToUi::Ready` (`bridge/protocol.rs`). Confirm what the bridge reports.
6. **Footer removal vs N1**: deleting the footer (Fix E cleaner path) must preserve the connection-status chip (`:610-621`) — it's the N1 "never a silent disconnect" surface. Fold it into row1's tail (Token::Error when disconnected) so a failed bridge stays visible. Don't drop it.
7. **Test churn**: Fixes A/G/H each flip an existing assertion (`:1121`, the title tests `:1632/1640`). These are deliberate spec reversals, not regressions — call them out in the PR so reviewers don't bounce them.
