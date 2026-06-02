# R4 Spinner/Cost Recon Spec

Cluster: spinner token arrows + Tip glyph + cost/ctx stats

---

## (1) REPRODUCED? — Rendered Evidence

**REPRODUCED** on the LIVE/styled path via `cockpit_rows_at` TestBackend.

The current busy-spinner row (from test `spinner_status_line_shape_with_thinking_effort`, width=100, max effort set, tok_in=1234 tok_out=340):

```
⠋ Razzmatazzing… (0.0s · ↓ 1.6k tokens · thinking with max effort)
⎿ <tip text>
```

Evidence from test `spinner_token_readout_reflects_live_counts`:
```
spinner contains: "↓ 1.6k tokens"
spinner does NOT contain "↑"
```

Tests explicitly **assert `!spinner.contains('↑')`** (footer.rs:374, 519), encoding the OLD requirement.

The **⎿** leader is present at footer.rs:195 (`"⎿ ".to_string()`) and the busy tip test at footer.rs:447 asserts `tip.starts_with("⎿ ")`.

The session_info row (from `session_info_shows_non_thinking_ctx_and_wire_identity`):
```
codex-pro  ·  gpt-5.5  ·  非思考模式  ·  ctx 48%  ·  <branch>
```
This shows `ctx 5%` in the user's actual session because the `context_percent` computed by `ga_bridge.py` uses a **char-based denominator** (`context_win * 3` chars) but the session likely has a very small history compared to the max context — so 5% is technically correct for the char-fill of the conversation. However, the denominator formula may over-inflate the cap, making the percentage look misleadingly small early in a conversation (correct value but surprising to user). The **cost** field (`cost_usd`) is wired in `AppState` (mod.rs:135) but **never shown in the session-info row** — `render_session_info` at footer.rs:150-177 shows llm · model · effort · ctx · branch, not cost.

---

## (2) ROOT CAUSE (exact file:line)

### A. Spinner shows only `↓ <out>` instead of `↑ <in> · ↓ <out>`

`src/components/cockpit/footer.rs:75-79` — `render_spinner` reads `app.tokens` (the combined total) for the down-arrow display. `app.tok_in` and `app.tok_out` (per-call last_input / last_output) are set via `reducer.rs:274-280` from the wire's `last_input`/`last_output` fields but **are never used by the spinner** — only by `render_done_line` (footer.rs:118-123).

```rust
// footer.rs:74-79 — CURRENT (only ↓ total)
if let Some(tokens) = app.tokens {
    g.push(Span::styled(" · ↓ ".to_string(), dim));
    g.push(Span::styled(human_count(tokens), text));
    g.push(Span::styled(format!(" {}", crate::i18n::t(app.lang, "tokens.unit")), dim));
}
```

The done-line (footer.rs:118-123) correctly uses both `tok_in` and `tok_out`:
```rust
// footer.rs:118-123 — CORRECT REFERENCE in render_done_line
if app.tok_in.is_some() || app.tok_out.is_some() {
    spans.push(Span::styled(" · ↑ ".to_string(), dim));
    spans.push(Span::styled(human_count(app.tok_in.unwrap_or(0)), text));
    spans.push(Span::styled(" · ↓ ".to_string(), dim));
    spans.push(Span::styled(human_count(app.tok_out.unwrap_or(0)), text));
}
```

The spinner must show BOTH arrows using `tok_in`/`tok_out` (from `last_input`/`last_output`) or fall back to `↓ tokens` if they are `None` (legacy bridge).

### B. No GRADUAL/animated counter increment

`app.tok_in` and `app.tok_out` are `Option<u64>` set directly from bridge wire (reducer.rs:275-280) — no easing. There is no `display_tok_in`/`display_tok_out` field. When the bridge emits a new Status (every ~1s during a turn, see ga_bridge.py:592), `tok_in`/`tok_out` jump immediately to the new value. `app.tokens` has the same problem. `app.tick()` (mod.rs:328-341) runs at 0.1s and could advance an eased display value.

CC's reference: `SpinnerAnimationRow.tsx:142-158` uses a `tokenCounterRef` that increments toward `currentResponseLength` by a gap-proportional step on each 50ms animation frame. GA bridge sends token updates ~1s apart so the jump is much coarser.

### C. `⎿` glyph instead of `└`

`src/components/cockpit/footer.rs:195`: `Span::styled("⎿ ".to_string(), dim)` — the ROW-2 tips.
The BUSY hanging tip is drawn via the SAME `render_tips` function (cockpit/mod.rs:215) so both use `⎿`. The comment in footer.rs:184 even says "restored from v2/v3" — but the user reports it was NOT changed to `└`. The file comment (line 2, mod.rs:10-11) shows `⎿` in the spec comment:
```
//!   SPINNER-TIP (1, only when busy)  ⎿ <rotating tip>
```
So `⎿` is the CURRENT glyph and must become `└`.

### D. `context_percent` formula: chars vs tokens

`ga_bridge.py:464-469` + `cost_tracker.py:45-62`:

```python
cap = ct.context_window_chars(be)      # = backend.context_win * 3
used = ct.current_input_chars(be)      # = sum(len(json.dumps(m)) for m in history)
frame["context_percent"] = used * 100.0 / cap
```

`context_win` is a GA-internal "chars" budget (e.g. 200000 chars = 200k). The ratio is char-based, not token-based. For a Claude/GPT backend with a 200k-char `context_win`, the cap is 600k chars. Early in a session with ~30k chars of history → 5%. This is internally consistent but confusing because the user expects token-based context % (like CC which shows "48%" when near context limit). The denominator is correct if `context_win` reflects the model's actual char budget; the issue is that early sessions genuinely are 5%.

**Probable true bug**: the `context_window_chars` function returns `context_win * 3`. If `context_win` is already measured in chars (not in tokens), then multiplying by 3 inflates the denominator 3×, making percentages show 3× too low. Example: if `context_win = 100000` chars, then cap = 300000 chars; a 30k-char history shows 10% instead of 30%. This is the likely "wrong" the user sees. The fix: check what unit `context_win` stores. If it stores chars, cap should be `context_win` not `context_win * 3`.

### E. Cost field not displayed

`render_session_info` (footer.rs:132-178) does not display `app.cost_usd`. The `cost_usd` field exists (mod.rs:135) but is only shown in the `/cost` overlay. No fix needed unless the spec calls for it — however since the user said "cost and context stats look wrong", this is worth speccing.

---

## (3) REFERENCE PATTERN

### A+B. CC `SpinnerAnimationRow.tsx:142-168` (claude-code):
```typescript
// Smooth token counter — increment toward target each 50ms frame
const gap = currentResponseLength - tokenCounterRef.current;
if (gap > 0) {
  let increment;
  if (gap < 70)        increment = 3;
  else if (gap < 200)  increment = Math.max(8, Math.ceil(gap * 0.15));
  else                 increment = 50;
  tokenCounterRef.current = Math.min(tokenCounterRef.current + increment, currentResponseLength);
}
const displayedResponseLength = tokenCounterRef.current;
const leaderTokens = Math.round(displayedResponseLength / 4); // chars→tokens estimate
// tokensText uses ↓ for solo leader, no ↓ for teammate aggregate
const tokensText = hasRunningTeammates ? `${tokenCount} tokens` : `↓ ${tokenCount} tokens`;
```

CC uses only `↓ <tokens>` (output token count derived from response chars). tui_v4 already has split input/output from the bridge and the done-line uses `↑ in · ↓ out`, which is a better pattern for the active spinner too.

### B. Reference format from `render_done_line` (footer.rs:118-123):
```
⠿ Razzmatazzing for 1m 46s · ↑ 1.2k · ↓ 340
```
The spinner should adopt the same `↑ <in> · ↓ <out>` format (without the "tokens" word to save width), identical to the done-line minus the duration.

### C. `└` vs `⎿`:
tuiapp_v2.py and tui_v3.py both use Python's box-drawing characters. The user-specified character `└` is U+2514 (BOX DRAWINGS LIGHT UP AND RIGHT). `⎿` is U+23BF (DENTISTRY SYMBOL LIGHT VERTICAL AND WAVE). The spec requires `└`.

### D. Context percent — tui_v3 reference (`tui_v3.py:2212-2224`):
```python
def _cost_str(agent) -> str:
    tot = sum(t.total_tokens for t in cost_tracker.all_trackers().values())
    ...
```
tui_v3 does NOT show context% in the status line at all — only in the `/cost` pane. So there's no v3 reference to correct it. CC shows context% as a bar in the footer. GA should check `context_win` units.

---

## (4) FIX SPEC

### 4.1 Spinner line: show `↑ <in> · ↓ <out>` (using `tok_in`/`tok_out`)

**File**: `src/components/cockpit/footer.rs`
**Function**: `render_spinner` (line 48–100)

Replace the token block inside `group(with_tokens: bool)` (currently lines 74–79) with a split-arrow block that reads `app.tok_in` and `app.tok_out`:

```rust
// NEW: inside the `group` closure, replace the tokens block
if with_tokens {
    let has_split = app.tok_in.is_some() || app.tok_out.is_some();
    if has_split {
        // Use display_tok_in/display_tok_out (eased) — see 4.2
        let di = app.display_tok_in.unwrap_or(app.tok_in.unwrap_or(0));
        let dout = app.display_tok_out.unwrap_or(app.tok_out.unwrap_or(0));
        g.push(Span::styled(" · ↑ ".to_string(), dim));
        g.push(Span::styled(human_count(di), text));
        g.push(Span::styled(" · ↓ ".to_string(), dim));
        g.push(Span::styled(human_count(dout), text));
    } else if let Some(tokens) = app.tokens {
        // Legacy fallback (bridge sent only total, no split)
        g.push(Span::styled(" · ↓ ".to_string(), dim));
        g.push(Span::styled(human_count(tokens), text));
        g.push(Span::styled(format!(" {}", crate::i18n::t(app.lang, "tokens.unit")), dim));
    }
}
```

Width gating: the `↑ <in> · ↓ <out>` segment is about 15–18 chars. Drop it first (same progressive gate that already exists at line 92–95).

### 4.2 Eased display values for tok_in / tok_out

**File**: `src/app/mod.rs`
Add two new fields to `AppState`:
```rust
/// Smoothly-animated DISPLAY values for the spinner's ↑/↓ readout.
/// Each tick() steps these toward the live tok_in/tok_out targets.
/// None until the first Status frame arrives (shows nothing until data exists).
pub display_tok_in: Option<u64>,
pub display_tok_out: Option<u64>,
```
Init both to `None` in `AppState::new()`.

**File**: `src/app/mod.rs`, function `tick()` (line 328)
Add at the end of `tick()`:
```rust
/// Ease display_tok_in/display_tok_out toward their live targets each tick (0.1s).
/// Mirrors CC SpinnerAnimationRow.tsx:142-158: gap-proportional step so small gaps
/// feel snappy and large jumps animate smoothly.
fn ease_tok(display: &mut Option<u64>, target: Option<u64>) {
    if let Some(t) = target {
        let d = display.get_or_insert(t);
        if *d < t {
            let gap = t - *d;
            let step = if gap < 70 { 3 } else if gap < 200 { (gap as f64 * 0.15).ceil() as u64 } else { 50 };
            *d = (*d + step).min(t);
        } else if *d > t {
            // Token counts should only go up during a turn; on a new turn reset instantly.
            *d = t;
        }
    } else {
        // No data yet: leave as None
    }
}
ease_tok(&mut self.display_tok_in, self.tok_in);
ease_tok(&mut self.display_tok_out, self.tok_out);
```

Reset `display_tok_in = None` and `display_tok_out = None` in `on_message_begin` (reducer.rs) so each new turn starts fresh (no carry-over from prior turn).

### 4.3 `⎿` → `└` glyph

**File**: `src/components/cockpit/footer.rs`
**Function**: `render_tips` (line 185–199)

Line 195: change `"⎿ "` to `"└ "`:
```rust
// BEFORE:
Span::styled("⎿ ".to_string(), dim),
// AFTER:
Span::styled("└ ".to_string(), dim),
```

This change covers BOTH call-sites automatically (the idle row2 below-composer tip AND the busy hanging spinner-tip both call the same `render_tips` function via cockpit/mod.rs:215 and 235).

Also update the cockpit/mod.rs doc comment at line 11 (`⎿ <rotating tip>` → `└ <rotating tip>`) for accuracy.

### 4.4 Context percent denominator fix

**File**: `D:/GenericAgent/frontends/cost_tracker.py`
**Function**: `context_window_chars` (line 45–52)

The comment says `context_win * 3` is the "char cap before `trim_messages_history` kicks in". Verify in the GA source (llmcore/trim_messages_history) whether the multiplier is correct. If `context_win` is measured in chars, the cap IS `context_win` not `context_win * 3`. The fix:

```python
def context_window_chars(backend) -> int:
    """The char cap for the message history (the trim trigger).
    GA's `trim_messages_history` trims when history chars exceed `context_win`.
    If `context_win` IS the char budget (common for llmcore), return it directly.
    The old `* 3` multiplied by 3x was wrong if context_win is already chars.
    """
    try:
        return int(getattr(backend, 'context_win', 0))   # NOT * 3
    except (TypeError, ValueError):
        return 0
```

**Important**: This change is in `D:/GenericAgent/frontends/cost_tracker.py` which IS editable (not GA core). Confirm the unit of `context_win` before changing. The investigator should run a quick test: during a session check `getattr(backend, 'context_win')` — if it prints e.g. `200000`, that's 200k CHARS, and the `* 3` was wrong. The denominator should be `context_win` directly.

If the `* 3` is correct (i.e., `context_win` is in tokens and char ≈ 3× tokens), then the formula is right and 5% is genuinely 5% — the user needs a better denominator, which should come from the model's actual context-window TOKENS reported in the Status frame. In that case, bridge.py should add a `context_tokens` field to the Status frame derived from a known model size (e.g. 200k tokens for claude-3-5), and the Rust side should add `app.context_capacity_tokens: Option<u64>`. This is a larger change but would be correct.

**Simpler immediate fix**: add model-based context capacity to `_status_payload` in `ga_bridge.py`:
```python
# In _status_payload, after computing context_percent, also try to get token-based cap
# from the model's known context window (tokenlimit on llmclient)
try:
    token_limit = getattr(getattr(self._agent, 'llmclient', None), 'token_limit', None)
    if token_limit:
        # token_limit is the model's context window in tokens; compute token-based %
        # by approximating from chars (4 chars ~ 1 token)
        used_tokens = used // 4  # rough
        frame["context_percent"] = float(min(100.0, used_tokens * 100.0 / token_limit))
except Exception:
    pass
```

### 4.5 Tests that MUST be updated

The following tests **encode the OLD requirement** (spinner shows only `↓ <total>`, no `↑`) and will FAIL after the fix — they MUST be rewritten:

1. **`footer.rs:363–380` — `spinner_token_readout_reflects_live_counts`**:
   - Line 373: `assert!(spinner.contains("↓ 1.6k tokens"), ...)` — will still pass
   - Line 374: `assert!(!spinner.contains('↑'), "no ↑in on the spinner line")` — **WILL FAIL** → remove or invert
   - Line 379: `assert!(spinner2.contains("↓ 61.2k tokens"), ...)` — may change format

2. **`footer.rs:477–525` — `spinner_status_line_shape_with_thinking_effort`**:
   - Line 515: `assert!(spinner.contains("· ↓ 1.6k tokens"), ...)` — format changes
   - Line 519: `assert!(!spinner.contains('↑'), "no ↑in")` — **WILL FAIL** → must invert

3. **`footer.rs:439–475` — `busy_tip_under_spinner_one_row_below_composer`**:
   - Line 447: `assert!(tip.starts_with("⎿ "), ...)` — **WILL FAIL** after ⎿→└
   - Line 474: `assert!(idle_rows[idle_bottom + 2].starts_with("⎿ "), ...)` — **WILL FAIL**

---

## (5) HONEST-CHECKS (exercises LIVE path, FAILS today, PASSES after fix)

### HC-1: Spinner shows both `↑ in` and `↓ out` (LIVE styled path)

```rust
#[test]
fn spinner_shows_both_up_and_down_arrows() {
    let rows = busy_spinner_rows(1234, 340);
    let spinner = find_spinner_row(&rows);
    // NEW requirement: BOTH arrows present
    assert!(spinner.contains("↑"), "spinner must contain ↑ input arrow: {spinner:?}");
    assert!(spinner.contains("↓"), "spinner must contain ↓ output arrow: {spinner:?}");
    // ↑ shows the input count (1234 → "1.2k")
    assert!(spinner.contains("↑ 1.2k"), "input token count: {spinner:?}");
    // ↓ shows the output count (340 → "340")
    assert!(spinner.contains("↓ 340"), "output token count: {spinner:?}");
}
```

**TODAY**: FAILS (↑ assertion fails — spinner has no ↑).
**AFTER FIX**: PASSES.

### HC-2: Tip rows use `└` leader, not `⎿` (LIVE styled path)

```rust
#[test]
fn tip_rows_use_floor_corner_glyph() {
    // Busy: hanging tip under spinner
    let rows = busy_spinner_rows(1234, 340);
    let spinner_idx = rows.iter().position(|r| r.contains('…') && r.contains('(')).unwrap();
    let tip = &rows[spinner_idx + 1];
    assert!(tip.starts_with("└ "), "busy hanging tip must use └: {tip:?}");
    assert!(!tip.starts_with("⎿"), "must NOT use ⎿: {tip:?}");

    // Idle: row2 tip below composer
    let mut idle = AppState::new();
    idle.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
    idle.model = Some("m".into());
    let idle_rows = cockpit_rows(&mut idle, 100, 30);
    let last = idle_rows[idle_rows.len() - 1].clone();
    assert!(last.starts_with("└ "), "idle tip row2 must use └: {last:?}");
    assert!(!last.starts_with("⎿"), "must NOT use ⎿: {last:?}");
}
```

**TODAY**: FAILS (rows start with "⎿ ").
**AFTER FIX**: PASSES.

### HC-3: Eased display values advance each tick (app.tick())

```rust
#[test]
fn display_tok_eases_toward_target_on_tick() {
    let mut app = AppState::new();
    // Seed tok_in/tok_out
    app.tok_in = Some(5000);
    app.tok_out = Some(1000);
    // display_tok* start as None
    assert!(app.display_tok_in.is_none());
    assert!(app.display_tok_out.is_none());
    // First tick: initializes to first step (gap=5000 → step=50)
    app.tick();
    let d_in = app.display_tok_in.expect("display_tok_in set after first tick");
    let d_out = app.display_tok_out.expect("display_tok_out set after first tick");
    assert!(d_in > 0 && d_in < 5000, "display_tok_in started easing: {d_in}");
    assert!(d_out > 0 && d_out < 1000, "display_tok_out started easing: {d_out}");
    // After enough ticks, converges to target
    for _ in 0..200 { app.tick(); }
    assert_eq!(app.display_tok_in, Some(5000), "converges to tok_in");
    assert_eq!(app.display_tok_out, Some(1000), "converges to tok_out");
}
```

**TODAY**: FAILS (`display_tok_in`/`display_tok_out` fields don't exist).
**AFTER FIX**: PASSES.

### HC-4: Width-gating for new `↑ · ↓` format drops tokens first on narrow terminal

```rust
#[test]
fn spinner_width_gating_drops_split_tokens_first() {
    use crate::app::effort::ReasoningEffort;
    let mut app = AppState::new();
    app.pet_style = crate::flavor::PetStyle::Off;
    app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
    app.model = Some("m".into());
    app.set_reasoning_effort(ReasoningEffort::Max);
    app.apply_bridge_event(
        BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
        0,
    );
    app.apply_bridge_event(
        BridgeEvent::Frame(CoreToUi::Status {
            model: None, llm: None, model_real: None,
            context_percent: Some(48.0), tokens: Some(1574),
            input_tokens: Some(1234), output_tokens: Some(340),
            cache_tokens: None, last_input: Some(1234), last_output: Some(340),
            text: None,
        }),
        0,
    );
    // Narrow width: drop tokens, keep elapsed + thinking phrase
    let rows = cockpit_rows_at(&mut app, 36, 12, 0);
    let spinner = find_spinner_row(&rows);
    assert!(!spinner.contains('↑'), "↑ tokens dropped on narrow: {spinner:?}");
    assert!(!spinner.contains('↓'), "↓ tokens dropped on narrow: {spinner:?}");
    assert!(spinner.contains("0.0s"), "elapsed kept: {spinner:?}");
    assert!(spinner.contains("thinking"), "thinking phrase kept: {spinner:?}");
}
```

**TODAY**: PASSES with OLD shape (only ↓ was present and gets dropped). After fix it should still pass (both arrows dropped).

---

## Summary

The active spinner status line currently shows only `↓ <combined total> tokens` using `app.tokens`, omitting the `↑ <input>` arrow entirely. The `tok_in`/`tok_out` fields (driven by `last_input`/`last_output` from the bridge) are available but only used by the frozen done-line. The `⎿` leader glyph in `render_tips` (footer.rs:195) must become `└` (U+2514). Context percent at 5% is produced by a char-based formula in `cost_tracker.py` that divides chars-used by `context_win * 3`; if `context_win` is already in chars the `* 3` inflates the denominator 3×, explaining the low value. Four tests currently encode the old behavior and must be rewritten: `spinner_token_readout_reflects_live_counts` (two assertions), `spinner_status_line_shape_with_thinking_effort` (two assertions), and `busy_tip_under_spinner_one_row_below_composer` (two assertions). No easing fields exist yet; `display_tok_in`/`display_tok_out` must be added to `AppState` and advanced in `tick()`.

VERDICT: REPRODUCED — spinner line shows only `↓ total` (missing `↑ input`), `⎿` tip glyph is unchanged, ctx% low due to `context_win * 3` over-denominator, no easing exists.
