# Round-4 fix spec — footer, spinner, pets, tab title, Tip line, per-command input FX

All paths relative to `frontends/tuiapp_v4/`. Each item gives the EXACT
`file:function` to change, the current (broken) behavior, and the precise fix.

---

## 1. SPINNER must be DYNAMIC (animate braille frames while busy, settle on all-dots when done)

**Where the glyph is chosen each frame:** `src/components/cockpit/footer.rs` →
`render_spinner()` (lines 39-90).

**Current (broken):**
- Line 41 computes `let tick = (now_ms / 100) as u64;` — a 0.1s frame counter.
- Line 42 HARDCODES `let glyph = '⠿';` and never uses `tick` to pick a frame, so
  the busy spinner is the STATIC all-dots glyph forever. The doc-comment at
  lines 36-38 even states this is deliberate ("the spinner glyph is a STATIC `⠿`
  … so we no longer index `spinner_style`"). That comment + the `spinner_emits_
  braille_all_dots` test (lines 309-326) are the spec that must be REVERSED.
- The animation machinery already exists and is unused here:
  `src/flavor/mod.rs::SpinnerStyle::glyph(frame)` (lines 60-64) cycles the frame
  set (`BRAILLE_FRAMES` = `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`, line 37), and `app.spinner_style`
  (`src/app/mod.rs:62`, default `SpinnerStyle::Arc`) carries the chosen style.

**Fix:**
- In `render_spinner` replace `let glyph = '⠿';` with an ANIMATED frame:
  `let glyph = app.spinner_style.glyph(tick);` (the busy spinner now cycles the
  active style's frames at the 0.1s `tick` rate while the turn runs).
- The "settle on all-dots when done" half is already correct and lives in a
  DIFFERENT function: `render_done_line()` (footer.rs:98-116) draws the frozen
  `⠿` (line 105) when idle-after-a-turn. The busy band (`render_spinner`) and the
  done band (`render_done_line`) are mutually exclusive via
  `split_cockpit` (`src/components/cockpit/mod.rs:106-109`: `show_spinner =
  app.busy`, `show_done = !show_spinner && last_turn_ms.is_some()`). So: keep
  `render_done_line`'s static `⠿`, animate ONLY `render_spinner`. Result =
  cycling braille while busy → settles on `⠿` when done, exactly as asked.
- Default style note: the all-dots-settle reads cleanest if the busy frames are
  the braille soul. Either flip `SpinnerStyle::default()` (flavor/mod.rs:26) from
  `Arc` to `Braille`, OR leave Arc as default (arc also animates correctly). The
  complaint only requires animation + an all-dots settle; `Braille` busy frames
  visually rhyme best with the `⠿` settle glyph — recommended.

**Tests to update (same file / module):**
- `footer.rs::spinner_emits_braille_all_dots` (lines 311-326): currently asserts
  the busy row carries the STATIC `⠿` and bans the arc set. Rewrite to assert the
  busy glyph is one of `app.spinner_style.frames()` and CHANGES across two
  different `now_ms` values (drive `cockpit_rows` at e.g. `now_ms=0` and
  `now_ms=300` and assert the spinner glyph differs). Keep the done-line's `⠿`
  assertion in `done_line_shows_elapsed_and_tokens_when_idle` (lines 226-279)
  unchanged.
- `flavor/mod.rs::spinner_wraps_cyclically` (lines 449-461) already covers
  `glyph()` cycling — no change needed.

---

## 2. /pets: REMOVE all spinner options; default pet = ON; default pet = bear

There is no `/pets` command today — the pet+spinner picker is `/emoji`. The
complaint's intent: the pet picker must stop offering SPINNER rows, and the pet
must default to BEAR (currently it defaults to OFF).

**Pet state + default:** `src/flavor/mod.rs::PetStyle` (lines 289-307). The
`#[default]` is on `Off` (lines 304-306), and the doc explicitly says "NOT emoji
pet by default". `src/app/mod.rs:64` is `pub pet_style: PetStyle` and
`AppState::new` derives it from `PetStyle::default()`.

**Picker rows (the "spinner config options"):** built in
`src/commands/dispatch.rs::emoji_picker_items()` (lines 552-578). Lines 567-577
append the 3 SPINNER rows (`SpinnerStyle::{Arc,Braille,Pulse}` at ids 100..=102).
The selection is applied in `apply_emoji_choice()` (lines 539-548), where lines
545-547 set `app.spinner_style` for ids 100..102.

**Fix:**
- Default pet = bear: in `flavor/mod.rs` move `#[default]` from the `Off` variant
  (lines 304-306) to the `Bear` variant (line 291). Update the variant doc-
  comments accordingly. This auto-flips `AppState::new`'s `pet_style` to `Bear`.
- Remove spinner config from the picker: in
  `dispatch.rs::emoji_picker_items` DELETE the spinner-row loop (lines 567-577);
  the picker becomes pet styles + Off only (ids 0..=5).
- In `dispatch.rs::apply_emoji_choice` DELETE the `else if (100..103)` arm
  (lines 545-547) so no row can set `spinner_style`. (Spinner style is now only a
  code default per item #1; it is no longer user-configurable from this picker.)
- The complaint names the command `/pets`. Either (a) rename the `/emoji`
  registry row (`src/commands/registry.rs:93`) `cmd("emoji", …)` to
  `cmd("pets", "pet style", Ui)` and update the dispatch arm key `"emoji"` in
  `dispatch.rs::open_ui_command` (line 135), or (b) add `/pets` as an `alias` of
  `emoji`. Recommended: rename to `pets` + add `alias("emoji", …, "pets")` so the
  old name still resolves. Update `registry.rs::registry_resolves_all_commands`
  (lines 322-339) name list + count, and `i18n` keys `emoji.*` (mod.rs:461-464,
  759-760) can stay or be renamed `pets.*`.

**Tests to update:**
- `flavor/mod.rs::pet_faces_are_five_styles_four_tiers_four_frames` (lines
  524-549): line 541 `assert_eq!(calm, "ʕ•ᴥ•ʔ")` still holds; add an assert that
  `PetStyle::default() == PetStyle::Bear`.
- Any test asserting `AppState::new().pet_style == PetStyle::Off` must flip to
  `Bear` (grep `pet_style` in `src/app/tests.rs`).
- `registry.rs` resolve test (item above).

---

## 3. Terminal TAB TITLE = dynamic-pet + session_name + GenericAgent (OSC), no "NativeClaude"

**Current title source:** `src/app/mod.rs::terminal_title()` (lines 429-433):
```rust
let bear = crate::flavor::PETS_BEAR[0][0]; // "ʕ•ᴥ•ʔ"
format!("{bear} GenericAgent · {model}")
```
- It already says `GenericAgent` and contains NO "NativeClaude" — the only
  "NativeClaude" strings in the tree are doc comments in `src/app/effort.rs:5,26`
  (about backend effort mapping); they do NOT touch the title. So the
  "must not contain NativeClaude" requirement is already met; do not regress it.
- TWO real gaps vs the spec: (a) the pet is a STATIC calm bear (`PETS_BEAR[0][0]`),
  not the DYNAMIC pet; (b) the title shows `model`, NOT the `session_name`.

**Where the title is emitted:** `src/app/mod.rs::sync_terminal_chrome()`
(lines 592-603) — called once per frame; writes via
`src/util/osc.rs::write_title` (lines 78-83, OSC 0 builder `build_title`
lines 58-64, control-char-stripped). Tab STATUS color is a separate OSC-21337
channel via `tab_status()` (lines 415-423) — leave that as-is.

**Fix (`terminal_title()`):**
- Use the DYNAMIC pet for the active style + current heat/blink frame:
  `let face = crate::flavor::pet(self.pet_style, self.turn_elapsed_ms(now_ms),
  self.spinner_tick);` — but `terminal_title()` takes no clock today and
  `sync_terminal_chrome` is the only caller. Two options:
  - Simplest/portable: title strings are coalesced (only re-emitted when CHANGED,
    lines 599-602), and many terminals truncate/clamp title repaint — a per-tick
    animated face would spam the emulator and most tabs only show a static title.
    So the pragmatic "dynamic-pet" reading = use the pet for the ACTIVE STYLE's
    representative face (heat-aware), NOT a per-0.1s blink. Recommended:
    `let face = crate::flavor::pet_face(self.pet_style, self.turn_elapsed_ms(0), 0);`
    falling back to the bear constant when `pet_style == Off`
    (so a user who turned the pet off still gets the bear identity the spec wants).
    With item #2 defaulting to Bear, this yields the bear out of the box.
  - If a truly animated face is wanted, thread `now_ms` into `terminal_title(now)`
    + `sync_terminal_chrome(now)` and gate the re-emit on a coarser cadence
    (e.g. heat-tier change) so it does not thrash.
- Swap `model` → `session_name`: the active session's name is
  `self.sessions.session(self.sessions.active).map(|s| s.name)` (the session map
  + `session_names.json` sidecar, `src/app/session.rs:779-826`). Build:
  ```rust
  let name = self.active_session_name(); // helper over sessions.active
  format!("{face} {name} · GenericAgent")
  ```
  (order: dynamic-pet + session_name + GenericAgent, per the complaint). If no
  name is set, fall back to `"GenericAgent"` only (avoid a bare `· GenericAgent`).
- Add a small `active_session_name(&self) -> &str` helper on `AppState` (or inline
  the lookup). `build_title` already strips control chars, so a stray name char is
  safe.

**Tests:** `src/app/tests.rs::tab_status_and_title_track_state` (lines 168-196)
asserts status transitions; extend with a title assert that
`terminal_title()` contains the active session name + "GenericAgent" + the pet
face, and does NOT contain "NativeClaude".

---

## 4. FOOTER row1: effort shows 非思考模式 when unset (not "—"); ctx must show a value

**Where row1 is formatted:** `src/components/cockpit/footer.rs::render_session_info()`
(lines 123-157).

### 4a. effort label
- Line 128: `let effort = app.effort_label().unwrap_or("—");`
- `app.effort_label()` (`src/app/mod.rs:352-354`) returns `None` until the user
  sets `/effort` (it maps `self.reasoning_effort` via `ReasoningEffort::label()`).
- **Fix:** when `None`, show the non-thinking label, not a dash. Add an i18n key
  `effort.none` and use it:
  - `src/i18n/mod.rs` EN table (near the emoji/effort keys, ~line 461): add
    `("effort.none", "non-thinking")`.
  - ZH table (~line 759): add `("effort.none", "非思考模式")`.
  - In `render_session_info` replace line 128 with:
    `let effort = app.effort_label().map(str::to_string)
        .unwrap_or_else(|| crate::i18n::t(app.lang, "effort.none").to_string());`
    and change the span at line 139 to take `effort` by value (it is currently
    `effort.to_string()` on a `&str`; with a `String` use `effort.clone()` or move).
- The SPINNER band has the same "no effort = nothing" gap (footer.rs:80-86 only
  pushes the effort token `if let Some(effort)`). For consistency with the Tip
  line spec (item #5, which wants "thinking with max effort" tokens), also render
  the non-thinking label there when `effort_label()` is `None`.

### 4b. ctx is blank (the real bug)
- Row1 ctx is formatted at footer.rs:129-132: `Some(p) => "ctx {p:.0}%"` else
  `"ctx —"`. The dash appears because `app.context_percent` is `None`.
- **Root cause:** there are TWO `on_status` reducers. The ACTIVE-session reducer
  (`src/app/reducer.rs:216-265`) DOES store it (line 233-236:
  `self.context_percent = Some(p)`). But the BACKGROUND-session reducer
  (`src/app/reducer.rs:344-361`) IGNORES it — its signature is
  `_context_percent: Option<f64>` (line 347) and the body only updates `model`.
  When the visible session's status frames are folded by the background reducer
  (or before the active session ever emits a `Status` with `context_percent`),
  `app.context_percent` stays `None` → "ctx —".
- **Fix:** make the background reducer store `context_percent` too (so a session
  promoted to active, or whose status is folded centrally, carries the value):
  in `reducer.rs:344-361` rename `_context_percent` → `context_percent` and add
  `if let Some(p) = context_percent { self.context_percent = Some(p);
  self.cost.context_percent = Some(p); }` mirroring the active reducer
  (lines 233-236). Verify which reducer the FOREGROUND session actually runs
  through (trace `apply_bridge_event` → `on_status` dispatch); if the active
  session already uses the rich reducer, the remaining blank is purely "no Status
  frame carried `context_percent` yet" — in that case ALSO confirm the bridge
  (`ga_bridge.py` Status emission) populates `context_percent`, and keep the
  `"ctx —"` fallback only for the genuine pre-first-status window.

**Tests:** `footer.rs` has `cockpit_rows` helpers seeding `context_percent:
Some(48.0)` (lines 238, 294) via the ACTIVE reducer — add a row1 assertion that
the rendered info row contains `ctx 48%` and (with no effort set) the non-thinking
label. Add a reducer unit test that the background `on_status` now stores
`context_percent`.

---

## 5. TIP position: corner-continuation (└/⎿) directly UNDER the spinner status line

**Current layout (detached at the very bottom):**
- `src/components/cockpit/mod.rs::split_cockpit()` (lines 101-173) lays the rows
  vertically as: header, sep, transcript(FLEX), [spinner XOR done], [dropdown],
  composer, **row1 info**, **row2 tips** — i.e. the Tip (`render_tips`,
  footer.rs:164-178, leader `⎿ `) is the LAST row, BELOW the composer, far from
  the spinner band which sits ABOVE the composer (lines 120-124).
- So today the spinner status line and the Tip are separated by the whole
  composer; the `⎿` reads as a footer leader, not a continuation of the spinner.

**Desired:** the Tip becomes a `└`/`⎿` continuation rendered immediately UNDER
the spinner's status line, where the status line is
`marker + gerund + ( elapsed · ↓ tokens · thinking <effort> )` (the
`render_spinner` line, footer.rs:52-89).

**Fix (layout move):**
- The Tip must move from the below-composer slot to a row attached to the spinner
  band. Two layout edits in `split_cockpit`:
  - When `show_spinner` is true, allocate the spinner as a TWO-row block (or push
    an extra `Constraint::Length(1)` for a `spinner_tip` row right after the
    spinner constraint at mod.rs:120-122) so the Tip sits on the line directly
    under the status line and ABOVE the composer.
  - Add a `pub spinner_tip: Option<Rect>` to `CockpitLayout` (lines 79-96),
    populated only when `show_spinner` (mirror the `spinner`/`done` Option wiring
    at lines 145-158), and in `render_cockpit` (mod.rs:192-194) draw
    `render_tips(frame, spinner_tip, …)` right after `render_spinner`.
  - Move the existing below-composer Tip OUT of the bottom: drop the row2 `tips`
    constraint (mod.rs:130) and the `render_tips(frame, tips, …)` call
    (mod.rs:211) for the BUSY case. When IDLE (no spinner), decide whether to keep
    a bottom Tip or hide it; the complaint only specifies the busy/continuation
    placement, so simplest is: Tip renders under the spinner when busy, and the
    old bottom row2 is removed (row1 session-info stays as the sole below-composer
    row). If a Tip is still wanted while idle, keep row2 but it is no longer the
    "under the spinner" one.
- The Tip leader glyph: `render_tips` already uses `⎿ ` (the rounded `└`,
  footer.rs:174). To read as a corner-CONTINUATION of the spinner status line,
  keep `⎿` and ensure the Tip row's left edge aligns under the spinner marker.
  No glyph change needed; only the POSITION changes.
- Compose the status line per the spec wording: ensure `render_spinner` emits
  `marker + gerund + (elapsed · ↓tokens · thinking <effort>)`. Today it emits
  marker(pet+glyph) + gerund + `(secs · ↑in ↓out · ctx … · effort)`
  (footer.rs:52-89). Add the "thinking <effort>" phrasing (prefix the effort
  token with the i18n "thinking" word; with `max` effort it reads "thinking max")
  and keep the down-arrow tokens. The `(…)` group already carries elapsed +
  arrows; only the effort token's label needs the "thinking" prefix.

**Tests:** `footer.rs::below_composer_has_two_rows` (lines 356-377) asserts
EXACTLY two rows follow the composer and row2 is `⎿`. This contract INVERTS:
update it so (busy) the `⎿` Tip appears ABOVE the composer immediately under the
spinner row, and below the composer there is now ONE row (session info). Add a
busy-state test that finds the spinner row (contains the spinner glyph) and
asserts the very NEXT row starts with `⎿ `.

---

## 6. COMMAND EFFECTS: /goal /hive /conductor /morphling restyle border + word like `!` does; DELETE /effects

The per-command FX engine already exists and is wired — the `!` mechanism it
mirrors is the template. The remaining work is to (a) confirm/round out the four
command identities and (b) delete the useless `/effects` command.

### The `!` template (what to copy), already implemented
- Detection: `src/input/shell.rs::is_shell_mode()` (lines 10-12) — first non-space
  char is `!`. Surfaced on the composer via `Composer::is_shell_mode()`
  (`src/input/mod.rs:159`).
- Border restyle: `src/components/cockpit/composer.rs::render_composer()`
  (lines 17-63): shell mode picks `Token::ShellAccent` for `border_tok` + `mark_tok`
  (lines 20-27) so the whole bordered block tints hot-pink.
- WORD/char restyle (the `!` peel): `composer.rs::composer_lines()` (lines 69-157)
  peels the leading `!` on row 0 and paints it `ShellAccent` (the `bang_pink`
  path, lines 102-105, 122-134), skipping the cursor cell.

### The command-FX mechanism (the parallel of `!`), already implemented
- Identity type: `src/theme/mod.rs::FxCommand` (lines 45-51) {Goal,Hive,Conductor,
  Morphling} and `FxCommand::border()` (lines 68-82) → an `FxBorder`
  (Goal=Pulse◆, Hive=Orbit⬡, Conductor=Sweep▸, Morphling=Rainbow◆), colors from
  theme tokens.
- Mapping: `src/components/text.rs::fx_command()` (lines 74-85) maps the leading
  `/word` to an `FxCommand` (rejects `/hivemind` etc.).
- Border paint: `src/components/effects_paint.rs::draw_composer_border_fx()`
  (lines 221-299) repaints the composer outline per the `FxBorder` motion +
  corner glyph; called from `render_cockpit` (`mod.rs:207-209`) ONLY when
  `fx_command(text).is_some()`, and it early-returns in shell mode (lines 231-233)
  so `!` keeps its plain hot-pink border.
- Char/word paint: `composer.rs::command_word_spans()` (lines 202-228) styles each
  char of the `/word` token per command (Morphling marches the rainbow, Hive
  swarms mint, Goal/Conductor sheen-sweep); peeled like the `!` via the `cmd_tok`
  path in `composer_lines` (lines 85, 110-129), skipping the cursor cell.

**So items in complaint #6 are ALREADY built.** The fix-spec work is:
- VERIFY the two paint paths fire together for all four words (border via
  `effects_paint`, word via `composer.rs`). One gap: the per-command BORDER paint
  is gated on `app.effects.caps.enabled()` (effects_paint.rs:228) — at mono/
  NO_COLOR the border won't restyle. The `!` border tint does NOT have that gate
  (it is a plain `border_style` color in `render_composer`). For parity ("exactly
  like `!`"), the four commands should ALSO tint the base `render_composer` border
  token even when the truecolor FX overlay is gated off. **Fix:** in
  `composer.rs::render_composer` extend the `border_tok`/`mark_tok` selection
  (lines 20-27) so that when `fx_command(text).is_some()` (and not shell), the
  base border token is the command's accent (e.g. map Goal/Morphling→Token::Claude,
  Hive→Token::Success, Conductor→Token::Suggestion — same sources `FxCommand::
  border` uses). This guarantees a visible border restyle for the four commands at
  every capability level, with the animated overlay layered on top when enabled —
  matching the always-on `!` border tint.
- DELETE `/effects` (the useless command):
  - `src/commands/registry.rs:98`: remove the `cmd("effects", …)` row.
  - Update `registry.rs::registry_resolves_all_commands` (lines 322-340): drop
    `"effects"` from the `all` list and the `COMMANDS.len()` count assertion.
  - `src/commands/dispatch.rs::app_command`: delete the entire `"effects"` arm
    (lines 309-335) including its `use effects::EffectMode;` local.
  - Remove now-unused effects-command i18n keys (`cmd.effects` and any
    demo/off/subtle/full helper strings) from `src/i18n/mod.rs`.
  - Keep the EFFECTS ENGINE itself (`src/effects/`, `effects_paint.rs`,
    `app.effects`, the `/effects demo` splash overlay machinery) — only the
    user-facing `/effects` COMMAND is removed; the border FX + separator shimmer
    continue to run automatically. (Decide a default `EffectMode` since the user
    can no longer set it — keep the current default, e.g. Subtle/Full, in
    `AppState::new`.)

**Tests:** `text.rs::fx_command_maps_four_words_rejects_hivemind` (lines 257-269)
and `composer.rs::command_word_spans_goal_returns_styled_spans` (lines 258-273)
already cover the word effect — keep. `registry.rs` resolve test must drop
`effects`. Add a `render_composer` test that the base border token differs from
`Token::Border` when the buffer holds `/goal` (border-restyle parity with `!`).

---

## Cross-cutting note
Items #1 (animate spinner) and #5 (Tip under spinner) both touch
`footer.rs::render_spinner` + the spinner-band layout in `cockpit/mod.rs`; do them
together. Items #2 and #6 both edit `commands/registry.rs` + `dispatch.rs`; do
them together. The "no NativeClaude" requirement (#3) is already satisfied — the
only matches are doc comments in `app/effort.rs`; do not introduce it elsewhere.
