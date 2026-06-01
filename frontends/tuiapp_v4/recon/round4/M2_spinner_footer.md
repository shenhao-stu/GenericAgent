# M2 — SPINNER / FOOTER / TIP / IDENTITY / TAB: Adversarial Render Verification

**Method**: `PYTHONUTF8=1 cargo run -q --manifest-path frontends/tuiapp_v4/Cargo.toml -- --dump-frame <scenario> 2>/dev/null`
Scenarios rendered: `busy`, `done`, `effort-high`, `normal`.
Source code cross-checked for every claim.

---

## Item 1 — SPINNER animates (braille frame, NOT all-dots) while busy; settles on `⠿` when done

**Status: CONFIRMED**

**Rendered evidence (busy frame, `now_ms=100`):**
```
⠙ Pondering… (0.1s · ↓ 1.6k tokens · non-thinking)
```
The glyph is `⠙` (BRAILLE_FRAMES[1], index = `tick % 10 = 1` at `now_ms=100`), NOT the all-dots `⠿`.

**Rendered evidence (done frame):**
```
⠿ Hooking for 1m 46s · ↑ 1.2k · ↓ 340
```
The done-line uses the static `⠿` (hardcoded in `render_done_line`, `footer.rs:114`).

**Code confirmation:**
- `flavor/mod.rs:38`: `BRAILLE_FRAMES = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']` — `⠿` is absent from the busy frame cycle.
- `flavor/mod.rs:29`: `SpinnerStyle::Braille` is `#[default]`.
- `footer.rs:51`: `let glyph = app.spinner_style.glyph(tick)` — animated from tick.
- `footer.rs:114`: `Span::styled("⠿ ".to_string(), ...)` — static, done-line only.
- Test `flavor/tests::braille_is_the_default_and_not_the_cc_asterisk` pins this.
- Test `footer/tests::spinner_animates_braille_frames` asserts `g0 != g3` across ticks.

**Adversarial probe**: At `now_ms=0` tick=0 → `⠋`, at `now_ms=300` tick=3 → `⠸`. Both are BRAILLE_FRAMES members, neither is `⠿`. The done-line is discriminated by `" for "` substring (it never appears in the busy spinner row).

---

## Item 2 — SPINNER STATUS LINE shape

**Claim:** `<braille> <gerund>… (elapsed · ↓ <count> tokens · thinking <effort>)`; no effort → non-thinking; with effort-high → "thinking with <level> effort"; token part reads `↓ <count> tokens`.

**Status: CONFIRMED**

**Rendered evidence (busy, no effort):**
```
⠙ Pondering… (0.1s · ↓ 1.6k tokens · non-thinking)
```

**Rendered evidence (effort-high scenario, medium effort applied):**
```
⠙ Pondering… (0.1s · ↓ 1.6k tokens · thinking with medium effort)
```

Shape breakdown:
- `⠙` = braille glyph (CONFIRMED)
- `Pondering…` = gerund + `…` (CONFIRMED)
- `(0.1s` = elapsed in seconds (CONFIRMED)
- `· ↓ 1.6k tokens` = down-arrow count "tokens" (CONFIRMED — uses `↓` not `↑`, and the i18n key `tokens.unit`="tokens")
- `· non-thinking)` = no-effort label from `i18n::thinking_phrase(effort=None)` → `t(lang, "effort.none")` = "non-thinking" (CONFIRMED)
- `· thinking with medium effort)` = with effort: `t("spinner.thinking") + " with " + level + " effort"` (CONFIRMED)

**Code confirmation:** `footer.rs:64–84` builds the group. `i18n/mod.rs:153–162` composes `thinking_phrase`. EN dict: `"effort.none" → "non-thinking"`, `"spinner.thinking" → "thinking"`.

**Note on "effort-high" scenario**: the dump seeds `ReasoningEffort::Medium`, then moves the slider +2 positions (to xhigh) but does NOT commit — so `app.reasoning_effort` stays `Medium` and the spinner correctly shows "thinking with medium effort", not "xhigh". This is correct behavior (spinner reflects committed effort, not the slider preview).

---

## Item 3 — HANGING TIP: `⎿ Tip:` sits DIRECTLY UNDER the spinner status line (above the composer) when busy

**Status: CONFIRMED**

**Rendered evidence (busy frame, row sequence):**
```
⠙ Pondering… (0.1s · ↓ 1.6k tokens · non-thinking)       ← spinner row
⎿ Tip: press / to open the command palette — ...           ← TIP row (directly under spinner)
╭──...──╮                                                  ← composer top border
│❯  type a message…                                        ← composer input
╰──...──╯                                                  ← composer bottom border
codex-pro  ·  gpt-5.5  ·  non-thinking  ·  ctx 48%  ·  — ← session info (1 row below composer)
```

The `⎿ Tip:` row is at `spinner_idx + 1` (immediately under the spinner), above the composer, confirming the "hanging tip" layout while busy.

**Idle layout comparison (normal frame):**
```
⠿ Pondering for 0s · ↑ 1.2k · ↓ 340                      ← done-line (above composer)
╭──...──╮
│❯  type a message…
╰──...──╯
codex-pro  ·  gpt-5.5  ·  non-thinking  ·  ctx 48%  ·  — ← session info (1 below)
⎿ Tip: press / ...                                         ← Tip row (2 below = row2)
```

Idle restores the tip to its "row2 below-composer" slot. No tip duplication.

**Code confirmation:** `footer.rs:185–199` renders the tip row. The cockpit layout (`cockpit/mod.rs`) assigns the tip area above the composer during busy (Slice 5 layout inversion). Test `footer/tests::busy_tip_under_spinner_one_row_below_composer` pins the exact index relationship.

---

## Item 4 — FOOTER row: `codex-pro · gpt-5.5 · (effort or 非思考模式) · ctx <pct>% · <branch>`

**Claim:** effort shows 非思考模式/non-thinking when unset (not a dash); ctx shows a value.

**Status: CONFIRMED**

**Rendered evidence (normal frame, effort=None):**
```
codex-pro  ·  gpt-5.5  ·  non-thinking  ·  ctx 48%  ·  —
```

**Rendered evidence (effort-high frame, effort=Medium):**
```
codex-pro  ·  gpt-5.5  ·  medium  ·  ctx 48%  ·  —
```

Fields confirmed:
- `codex-pro` = llm name from `app.llm_name` (CONFIRMED, seeded via Status frame)
- `gpt-5.5` = model_real from `app.model_real` (CONFIRMED)
- `non-thinking` = `t(lang, "effort.none")` when `effort_label()=None` — NOT a dash (CONFIRMED)
- `medium` = effort label when set (CONFIRMED)
- `ctx 48%` = context percent from Status frame (CONFIRMED, value present not "—")
- `—` = git branch (no branch discovered in dump-frame, fallback to `app.git_branch=None` → "—") (CONFIRMED)

**Code confirmation:** `footer.rs:146–149`: `effort = app.effort_label().map(...).unwrap_or_else(|| t(lang, "effort.none"))`. No dash fallback — always shows the non-thinking label.

---

## Item 5 — TAB TITLE: dynamic pet face + session_name + GenericAgent, default bear, NO "NativeClaude"

**Status: CONFIRMED**

**Code evidence (`app/mod.rs:460–471`):**
```rust
pub fn terminal_title(&self) -> String {
    let face = match crate::flavor::pet_face(self.pet_style, ..., 0) {
        "" => crate::flavor::PETS_BEAR[0][0],  // pet Off → keep the bear identity.
        f => f,
    };
    let name = self.sessions.active_name();
    if name.is_empty() {
        format!("{face} GenericAgent")
    } else {
        format!("{face} {name} · GenericAgent")
    }
}
```

- Default pet style: `PetStyle::Bear` (`flavor/mod.rs:29`: `#[default]`).
- Bear face at idle, calm tier, frame 0: `PETS_BEAR[0][0]` = `"ʕ•ᴥ•ʔ"`.
- Session name present → `"ʕ•ᴥ•ʔ session 1 · GenericAgent"`.
- `PetStyle::Off` fallback keeps `PETS_BEAR[0][0]` (never an empty title).
- String `"NativeClaude"` appears NOWHERE in any title string literal; only in comments and test assertions that explicitly assert its absence.

**Test confirmation (`app/tests.rs:241–268`):**
```rust
assert!(title.starts_with("ʕ•ᴥ•ʔ"), ...);
assert!(title.contains("GenericAgent"), ...);
assert!(!title.contains("NativeClaude"), ...);
assert_eq!(renamed, "ʕ•ᴥ•ʔ scan tabs · GenericAgent");
```

**Adversarial grep**: `grep -rn "NativeClaude" src/` finds only two test assertions and one comment — never a string that would appear in the rendered title.

---

## Summary

| Item | Claim | Status | Key Evidence |
|------|-------|--------|-------------|
| 1 | Spinner animates (braille) while busy; settles on `⠿` when done | **CONFIRMED** | `⠙` at busy frame; `⠿` at done frame; `⠿` not in BRAILLE_FRAMES |
| 2 | Status line shape `<braille> <gerund>… (elapsed · ↓ <count> tokens · thinking <effort>)` | **CONFIRMED** | `⠙ Pondering… (0.1s · ↓ 1.6k tokens · non-thinking)` exact match |
| 3 | `⎿ Tip:` directly under spinner (above composer) when busy | **CONFIRMED** | spinner_idx+1 = Tip row; composer follows; idle restores to row2 |
| 4 | Footer: `llm · model · effort/non-thinking · ctx <pct>% · branch` | **CONFIRMED** | `codex-pro  ·  gpt-5.5  ·  non-thinking  ·  ctx 48%  ·  —` |
| 5 | Tab title: `<bear-face> <session_name> · GenericAgent`, no "NativeClaude" | **CONFIRMED** | `terminal_title()` code; test pins `"ʕ•ᴥ•ʔ scan tabs · GenericAgent"` |

**No GAPs found.** All five claims verified by rendered frame output and source code.
