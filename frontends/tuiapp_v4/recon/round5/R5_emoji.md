# R5_emoji — Unify /emoji + animate tab + remove /pets

## 1. REPRODUCED?

YES — two distinct bugs confirmed via live-path inspection.

### Bug A: terminal-title emoji is STATIC (never animates)

`terminal_title()` in `src/app/mod.rs:461` calls:

```rust
crate::flavor::pet_face(self.pet_style, self.turn_elapsed_ms(0), 0)
//                                                               ^
//                                        frame is HARDCODED to 0 — never changes
```

The frame argument is always `0`. Because `pet_face(style, elapsed, 0)` always returns the
`pool[tier][0 % 4]` entry, the face never advances to frames 1/2/3. The title is therefore
pinned to the calm-tier-frame-0 glyph (`ʕ•ᴥ•ʔ` for Bear) regardless of time.

The idempotence guard in `sync_terminal_chrome()` (`if self.last_title != title`) means the
OSC 0 write happens exactly once at startup and then never again, so the terminal tab title
is completely frozen for the life of the session.

**Rendered evidence** (what `terminal_title()` returns at tick 0 vs tick 300 with Bear
default):

```
tick  0: "ʕ•ᴥ•ʔ GenericAgent"
tick 30: "ʕ•ᴥ•ʔ GenericAgent"   ← SAME — should be "ʕ-ᴥ-ʔ GenericAgent"
```

`pet_face(Bear, 0, 30/5)` = `pet_face(Bear, 0, 6 % 4)` = `PETS_BEAR[0][2]` = `"ʕ•ᴥ•ʔ"` (still
frame 2 wraps back to `•ᴥ•` in that tier), but the key point is the divisor `/5` is also
missing — the frame argument is not `self.spinner_tick / PET_TICKS_PER_FRAME`, it is `0`.

### Bug B: /pets command exists as a separate command / /emoji is only an alias

`src/commands/registry.rs:93-94`:

```rust
cmd("pets", "pet style", Ui),
alias("emoji", "pet style", Ui, "pets"),
```

`/pets` is the PRIMARY command; `/emoji` is merely an alias of it. The picker opened by both
commands (`emoji_picker_items`) lists ONLY the 5 pet styles + Off — it does NOT offer
braille / arc / pulse as selectable spinner glyphs. Selecting "bear" only changes
`app.pet_style`; it does NOT change `app.spinner_style` (which drives `render_spinner`'s
lead glyph). The spinner lead glyph is ALWAYS `app.spinner_style.glyph(tick)` regardless of
pet selection.

The user's intent: bear/cat/… and ⠿/braille/arc/pulse should be ONE mutually-exclusive pick
under `/emoji`; picking "bear" must drive the spinner glyph in the footer AND the tab title.
`/pets` should not exist at all.

---

## 2. ROOT CAUSE

### Root cause A — static frame in terminal_title

File: `src/app/mod.rs`, function `terminal_title()`, line 461.

```rust
// CURRENT (broken):
let face = match crate::flavor::pet_face(self.pet_style, self.turn_elapsed_ms(0), 0) {
```

The third argument `0` is hardcoded. It must be `self.spinner_tick / crate::flavor::PET_TICKS_PER_FRAME`.
Additionally, `terminal_title` takes `&self` but has no access to `now_ms`; it uses
`self.spinner_tick` (a `u64` field advanced each `tick()` call) which IS available on
`&self`. The PET_TICKS_PER_FRAME divisor (`5`) already exists in `src/flavor/mod.rs:418` as
a public constant.

### Root cause B — /pets is the primary; unified Companion enum missing

File: `src/commands/registry.rs`, lines 93-94 — `/pets` is `cmd(...)` (primary) and
`/emoji` is `alias(...)`. The fix inverts this: make `/emoji` the primary `cmd(...)` and
`/pets` simply disappears (removed from the table entirely).

The picker items function `emoji_picker_items` in `src/commands/dispatch.rs:537-554` builds
rows only from `PetStyle::all()` + Off. It must be extended to include the three spinner
styles (braille/arc/pulse) as additional pick-one items.

The `apply_emoji_choice` function at `src/commands/dispatch.rs:528-534` only sets
`app.pet_style`; it must additionally set `app.spinner_style` (for spinner rows) and set
`app.pet_style = PetStyle::Off` (for spinner rows) so the two fields stay coherent.

There is no unified "Companion" enum yet. The cleanest approach is a **virtual compound
enum** (a selection tag) stored in `AppState` that maps to the underlying
`SpinnerStyle`/`PetStyle` pair. This avoids restructuring `flavor/mod.rs` while keeping the
picker logic cohesive.

---

## 3. REFERENCE PATTERN

### tui_v3.py — how it does /emoji + spinner title

`tui_v3.py:1902-1911`: `_PET_STYLES` dict contains only pet-face tables (bear/cat/dot/unicode).
The braille spinner (`_SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'`) is NOT in the picker — in tui_v3 the
pet is only in the spinner line (not in the tab title).

`tui_v3.py:4771-4775` — `_term_title()`:
```python
def _term_title(self) -> str:
    name = (self._session_name or '').strip()
    head = (_SPIN[self._spin % len(_SPIN)] + ' ') if self._running else ''
    tail = f'{name} · GenericAgent' if name else 'GenericAgent'
    return f'{head}{tail}'
```
tui_v3 uses the SPINNER glyph (braille character) as the title prefix while running, and
calls `_set_term_title(self._term_title())` on every 0.1s tick (`tui_v3.py:4769`). The pet
face is ONLY in the spinner status line, not the title.

`tui_v3.py:3289-3297` — spinner line rendering:
```python
after.append(' ' + _heat(el) + _pet(el, self._spin // 5) + _RST +
             '  ' + _DIM + _gerund(el) + '…' + _RST)
```
The pet face is the LEAD of the spinner line with `_spin // 5` for the ~0.5s cadence.

**tui_v3's design intent** (the reference the spec targets): the selected "emoji/pet" style
drives the spinner line's lead glyph AND the tab title animates. tui_v3 couples both.

### tui_v3.py — /emoji is the ONE command (no /pets)

`tui_v3.py:4507-4508`: `elif name == 'emoji': self._cmd_emoji(arg)` — only `/emoji`, no
`/pets`. The picker includes ALL pet-face styles (bear/cat/dot/unicode) plus off. The braille
spinner is the hardcoded spinning glyph tui_v3 ALWAYS uses; it is not offered as a picker
choice in tui_v3. tui_v4 adds arc/pulse as additional options beyond tui_v3.

### Codex (src in D:/GenericAgent/temp/codex_src/codex-rs)

Not used as a reference for this specific feature — codex uses its own spinner model without
a "pet" concept. tui_v3 is the authoritative reference here.

---

## 4. FIX SPEC

### 4a. Introduce a CompanionKind selector (src/flavor/mod.rs)

Add a new enum `CompanionKind` (no struct, just a tagged union of the two existing enum
values) to represent the user's unified /emoji selection:

```rust
/// The unified /emoji selection: a spinner style (braille/arc/pulse)
/// OR a pet style (bear/cat/dot/unicode/fox/off). These are mutually
/// exclusive — picking one variant sets the other to its neutral value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompanionKind {
    Spinner(SpinnerStyle),
    Pet(PetStyle),
}

impl Default for CompanionKind {
    fn default() -> Self {
        CompanionKind::Pet(PetStyle::Bear)
    }
}

impl CompanionKind {
    /// Returns the effective SpinnerStyle to use in render_spinner.
    /// Pet variants fall back to SpinnerStyle::Braille (the tui_v3 soul).
    pub fn spinner_style(self) -> SpinnerStyle {
        match self {
            CompanionKind::Spinner(s) => s,
            CompanionKind::Pet(_) => SpinnerStyle::Braille,
        }
    }

    /// Returns the effective PetStyle for the tab title face.
    /// Spinner variants use PetStyle::Off (no face in the tab title).
    pub fn pet_style(self) -> PetStyle {
        match self {
            CompanionKind::Pet(p) => p,
            CompanionKind::Spinner(_) => PetStyle::Off,
        }
    }

    /// The lead glyph for the spinner status line at a given tick.
    /// For a pet variant this IS the pet face (at the /5 cadence),
    /// for a spinner variant it is the animated spinner character.
    pub fn spinner_lead(self, elapsed_ms: u64, tick: u64) -> String {
        match self {
            CompanionKind::Spinner(s) => s.glyph(tick).to_string(),
            CompanionKind::Pet(p) => {
                let face = pet_face(p, elapsed_ms, tick / PET_TICKS_PER_FRAME);
                if face.is_empty() {
                    SpinnerStyle::Braille.glyph(tick).to_string()
                } else {
                    face.to_string()
                }
            }
        }
    }

    /// The tab-title face for the current tick + elapsed_ms.
    /// For a pet variant: the animated face (tick / 5).
    /// For a spinner variant: the animated spinner glyph.
    pub fn title_face(self, elapsed_ms: u64, tick: u64) -> &'static str {
        match self {
            CompanionKind::Pet(PetStyle::Off) => PETS_BEAR[0][0], // fallback always shows bear
            CompanionKind::Pet(p) => pet_face(p, elapsed_ms, tick / PET_TICKS_PER_FRAME),
            CompanionKind::Spinner(s) => {
                // Use a static char; we need &'static str — use glyph index into the frame
                // array directly. Since SpinnerStyle::frames() returns Vec<char>, we must
                // return from the static slice arrays.
                let frames: &'static [char] = match s {
                    SpinnerStyle::Arc => ARC_FRAMES,
                    SpinnerStyle::Braille => BRAILLE_FRAMES,
                    SpinnerStyle::Pulse => PULSE_GLYPHS,
                };
                // SAFETY: char is 4 bytes max; use the index into static frame slice.
                // Return a &'static str of the char — we need an intermediate static buffer.
                // Simplest: return the PET_HIDDEN marker as a sentinel and let terminal_title
                // do a runtime format! — see note in 4c below.
                let _ = frames; // see 4c for the actual title_face API
                PETS_BEAR[0][0] // placeholder — see 4c for the correct impl
            }
        }
    }
    
    /// Display name for the picker.
    pub fn display_name(self) -> &'static str {
        match self {
            CompanionKind::Spinner(s) => s.name(),
            CompanionKind::Pet(p) => p.name(),
        }
    }
}
```

**Note on &'static str for spinner glyphs**: `SpinnerStyle::glyph()` returns `char`, not
`&'static str`. The `terminal_title()` caller does a `format!` anyway, so the cleanest
approach is to change `terminal_title`'s face binding to a `String` rather than `&'static str`,
using `companion.spinner_lead(elapsed, tick)` which already returns `String`. See 4c.

### 4b. Replace the two separate fields with CompanionKind in AppState (src/app/mod.rs)

**Fields to change** (lines ~72-76 in `src/app/mod.rs`):

```rust
// REMOVE:
pub spinner_style: SpinnerStyle,
pub pet_style: PetStyle,

// ADD:
/// The unified /emoji companion selection (spinner glyph or pet face).
/// Drives both the spinner lead glyph and the tab-title face.
pub companion: CompanionKind,
```

Update `AppState::new()` constructor to initialize `companion: CompanionKind::default()`.

All existing callers of `app.spinner_style` and `app.pet_style` must be migrated:
- `src/components/cockpit/footer.rs:51` — `app.spinner_style.glyph(tick)` becomes
  `app.companion.spinner_lead(elapsed, tick)` (as a `String`; adjust the Span format).
- `src/app/mod.rs:461` — see 4c below.
- `src/commands/dispatch.rs:529-533` — see 4d below.
- `src/commands/dispatch.rs:545-549` — see 4d below.

**Tests in `src/components/cockpit/footer.rs`** reference `app.pet_style = PetStyle::Off`
(line ~308) and `SpinnerStyle` imports (line ~207). These need updating to use
`app.companion = CompanionKind::Spinner(SpinnerStyle::Braille)` or similar — the existing
`spinner_animates_braille_frames` test's `busy_spinner_rows_at` helper sets `pet_style Off`
so the lead is bare glyph; that intent becomes `companion = CompanionKind::Spinner(SpinnerStyle::Braille)`.

### 4c. Fix terminal_title to use spinner_tick (src/app/mod.rs:460-471)

```rust
// FIXED terminal_title:
pub fn terminal_title(&self) -> String {
    let tick = self.spinner_tick;
    let elapsed = self.turn_elapsed_ms(0);
    let face: String = match self.companion {
        CompanionKind::Pet(PetStyle::Off) => {
            crate::flavor::PETS_BEAR[0][0].to_string() // Off → keep bear identity
        }
        CompanionKind::Pet(p) => {
            crate::flavor::pet_face(p, elapsed, tick / crate::flavor::PET_TICKS_PER_FRAME)
                .to_string()
        }
        CompanionKind::Spinner(s) => {
            s.glyph(tick).to_string()
        }
    };
    let name = self.sessions.active_name();
    if name.is_empty() {
        format!("{face} GenericAgent")
    } else {
        format!("{face} {name} · GenericAgent")
    }
}
```

Key change: `pet_face(..., tick / PET_TICKS_PER_FRAME)` instead of `pet_face(..., 0)`. The
`spinner_tick` field is already in `&self` and is advanced every 0.1s by `tick()`. Since
`sync_terminal_chrome()` is called after every frame draw (including tick-driven redraws),
and because `terminal_title()` now returns different strings at different tick values, the
idempotence guard `if self.last_title != title` will fire on every tick where the face
advances (every 5 ticks = every 0.5s for pets, every tick for spinner glyphs). This matches
tui_v3's behaviour exactly (it calls `_set_term_title(self._term_title())` every 0.1s).

### 4d. Unify the picker — /emoji is the primary, /pets is removed (src/commands/registry.rs + dispatch.rs)

**registry.rs** (lines 93-94):

```rust
// REMOVE both lines:
cmd("pets", "pet style", Ui),
alias("emoji", "pet style", Ui, "pets"),

// ADD one line:
cmd("emoji", "spinner or pet companion style", Ui),
```

Also update the `all_command_names` string array (line ~326) to remove `"pets"` and keep
only `"emoji"`.

**dispatch.rs** — rename the dispatch arm:

```rust
// CHANGE "pets" arm name to "emoji":
"emoji" => {
    let items = emoji_picker_items(app);
    let backup = app.theme.clone();
    app.open_picker(Picker::new(PickerKind::Emoji, items), Some(backup));
}
```

**dispatch.rs — `emoji_picker_items`**: expand to include spinner rows with ids ≥ 100:

```rust
pub(crate) fn emoji_picker_items(app: &AppState) -> Vec<PickItem> {
    use flavor::{PetStyle, SpinnerStyle, CompanionKind};
    let mut items: Vec<PickItem> = Vec::new();

    // Section 1: spinner glyphs (ids 100/101/102) — braille first (default).
    for (id_offset, style) in [
        SpinnerStyle::Braille,
        SpinnerStyle::Arc,
        SpinnerStyle::Pulse,
    ].iter().enumerate() {
        let id = 100 + id_offset;
        let sample = style.glyph(0).to_string();
        let is_current = app.companion == CompanionKind::Spinner(*style);
        items.push(
            PickItem::new(id, format!("spinner · {}", style.name()))
                .with_detail(sample)
                .current(is_current),
        );
    }

    // Section 2: pet faces (ids 0..=4) — bear/cat/dot/unicode/fox.
    for (i, style) in PetStyle::all().iter().enumerate() {
        let is_current = app.companion == CompanionKind::Pet(*style);
        items.push(
            PickItem::new(i, format!("pet · {}", style.name()))
                .with_detail(flavor::pet_face(*style, 0, 0).to_string())
                .current(is_current),
        );
    }

    // Section 3: pet off (id 5).
    let is_off = app.companion == CompanionKind::Pet(PetStyle::Off);
    items.push(PickItem::new(5, "pet · off".to_string()).current(is_off));

    items
}
```

**dispatch.rs — `apply_emoji_choice`**: handle both spinner and pet ids:

```rust
pub(crate) fn apply_emoji_choice(app: &mut AppState, id: usize) {
    use flavor::{PetStyle, SpinnerStyle, CompanionKind};
    app.companion = match id {
        100 => CompanionKind::Spinner(SpinnerStyle::Braille),
        101 => CompanionKind::Spinner(SpinnerStyle::Arc),
        102 => CompanionKind::Spinner(SpinnerStyle::Pulse),
        0..=4 => CompanionKind::Pet(PetStyle::all()[id]),
        5 => CompanionKind::Pet(PetStyle::Off),
        _ => return, // unknown id — no change
    };
}
```

### 4e. Update render_spinner to use companion.spinner_lead (src/components/cockpit/footer.rs)

```rust
// CURRENT line 51:
let glyph = app.spinner_style.glyph(tick);

// FIXED:
let lead_str = app.companion.spinner_lead(elapsed, tick);

// CURRENT lines 60-63:
let lead = vec![
    Span::styled(format!("{glyph} "), heat_style),
    ...
];

// FIXED (lead_str is a String):
let lead = vec![
    Span::styled(format!("{lead_str} "), heat_style),
    ...
];
```

### 4f. Remove the now-wrong tests and update honest-check test (src/commands/dispatch.rs)

The existing `pets_picker_has_zero_spinner_rows_and_bear_title` test ACTIVELY ASSERTS the
old (broken) state — it checks that `/pets` resolves, that `/emoji` is an alias of `/pets`,
and that there are zero spinner rows. After the fix this test must be DELETED and REPLACED
with the new honest-check (see section 5).

Also update `aliases_marked_not_duplicated` in `src/commands/registry.rs` to remove the
`("emoji", "pets")` entry.

---

## 5. HONEST-CHECK

The following test exercises the LIVE path (AppState + apply_bridge_event + cockpit render)
and would FAIL today, PASS after the fix. It should live in
`src/commands/dispatch.rs` (tests module) replacing the old
`pets_picker_has_zero_spinner_rows_and_bear_title`.

```rust
/// ROUND-5 HONEST CHECK: unified /emoji command.
/// (a) /pets no longer resolves; /emoji is the primary command (not an alias).
/// (b) The picker has BOTH spinner rows (braille/arc/pulse, ids 100-102) and pet
///     rows (bear/cat/…, ids 0-4) + off (id 5) — 9 rows total.
/// (c) Selecting "bear" drives app.companion = Pet(Bear), which:
///     - causes render_spinner's lead glyph to be the bear face (not a braille char)
///     - causes terminal_title to ANIMATE across tick values (face differs at tick 0 vs tick 5)
/// (d) terminal_title at tick 5 differs from tick 0 (animation proof).
/// (e) A spinner selection (braille) reverts the spinner lead to braille glyphs.
#[test]
fn emoji_unified_picker_and_animated_title() {
    use crate::app::AppState;
    use crate::bridge::{protocol::CoreToUi, BridgeEvent};
    use crate::commands::{dispatch::*, resolve};
    use crate::flavor::{CompanionKind, PetStyle, SpinnerStyle, BRAILLE_FRAMES, PET_TICKS_PER_FRAME};
    use crate::components::cockpit::footer::tests::{cockpit_rows_at, find_spinner_row};

    // (a) /pets is GONE; /emoji is a primary cmd (no alias_of).
    assert!(resolve("pets").is_none(), "/pets must no longer resolve after fix");
    let emoji_cmd = resolve("emoji").expect("/emoji must resolve");
    assert!(emoji_cmd.alias_of.is_none(), "/emoji must be a primary command, not an alias");

    // (b) The picker has 9 rows: 3 spinner + 5 pet + 1 off.
    let app = AppState::new();
    let items = emoji_picker_items(&app);
    assert_eq!(items.len(), 9, "3 spinners + 5 pets + off = 9 rows, got {:?}",
        items.iter().map(|i| (i.id, i.label.clone())).collect::<Vec<_>>());
    let spinner_rows: Vec<_> = items.iter().filter(|i| i.id >= 100).collect();
    assert_eq!(spinner_rows.len(), 3, "3 spinner rows (ids 100/101/102)");
    for label in ["braille", "arc", "pulse"] {
        assert!(spinner_rows.iter().any(|r| r.label.contains(label)),
            "spinner row for {label} missing");
    }

    // (c) bear selection — companion = Pet(Bear) — spinner lead is the bear face.
    let mut app = AppState::new();
    // Open a busy turn so the spinner line is rendered.
    app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
    app.model = Some("m".into());
    app.apply_bridge_event(
        BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
        0,
    );
    apply_emoji_choice(&mut app, 0); // id 0 = Bear
    assert_eq!(app.companion, CompanionKind::Pet(PetStyle::Bear),
        "selecting id=0 must set companion to Pet(Bear)");
    // At tick 0 the bear face is PETS_BEAR[calm_tier][0] = "ʕ•ᴥ•ʔ".
    let rows0 = cockpit_rows_at(&mut app, 100, 30, 0);
    let spinner0 = find_spinner_row(&rows0);
    assert!(spinner0.contains("ʕ•ᴥ•ʔ"),
        "spinner lead must be the bear face at tick 0, got: {spinner0:?}");

    // (d) terminal_title ANIMATES — tick 0 face differs from tick 5.
    app.spinner_tick = 0;
    let title0 = app.terminal_title();
    app.spinner_tick = PET_TICKS_PER_FRAME; // = 5 ticks → frame 1 for the pet
    let title5 = app.terminal_title();
    assert_ne!(title0, title5,
        "terminal_title must differ at tick 0 vs tick 5 (face animates): t0={title0:?} t5={title5:?}");

    // (e) braille selection — spinner lead is braille glyph, NOT a pet face.
    apply_emoji_choice(&mut app, 100); // id 100 = Braille spinner
    assert_eq!(app.companion, CompanionKind::Spinner(SpinnerStyle::Braille));
    // Reset tick so tick = 0, glyph = BRAILLE_FRAMES[0] = '⠋'.
    app.spinner_tick = 0;
    let rows_b = cockpit_rows_at(&mut app, 100, 30, 0);
    let spinner_b = find_spinner_row(&rows_b);
    assert!(spinner_b.starts_with(BRAILLE_FRAMES[0].to_string().as_str()),
        "braille selection must restore the braille lead glyph: {spinner_b:?}");
    // And the title at tick 5 for braille is the glyph at frame 5.
    app.spinner_tick = 5;
    let title_b5 = app.terminal_title();
    let expected_glyph = SpinnerStyle::Braille.glyph(5);
    assert!(title_b5.starts_with(expected_glyph.to_string().as_str()),
        "title with braille companion must use the braille glyph at tick 5: {title_b5:?}");
}
```

The test exercises the live-styled path via `cockpit_rows_at` (which calls
`apply_bridge_event` then `Terminal::new(TestBackend::new(...))` and renders with real
`render()`) — not a headless dump, not a plain-text fake.

**Why it FAILS today**:
- `resolve("pets")` returns `Some(...)` → assertion `pets must no longer resolve` fails.
- `emoji_picker_items` returns 6 rows (pet only) → `items.len() == 9` fails.
- `app.companion` field does not exist → compile error.
- `app.terminal_title()` at tick 0 and tick 5 returns the same string → `assert_ne` fails.

**Why it PASSES after the fix**:
- `/pets` is removed from the registry.
- `emoji_picker_items` returns 3 spinner + 5 pet + 1 off = 9 rows.
- `apply_emoji_choice(0)` sets `app.companion = CompanionKind::Pet(Bear)`.
- `cockpit_rows_at` renders the bear face as the lead.
- `terminal_title` at tick 5 returns frame 1 of the calm-tier bear pool (`ʕ-ᴥ-ʔ`), which
  differs from tick 0 (`ʕ•ᴥ•ʔ`).

---

## Summary of files to change

| File | Change |
|------|--------|
| `src/flavor/mod.rs` | Add `CompanionKind` enum with `spinner_style()`, `pet_style()`, `spinner_lead()`, `title_face()` methods |
| `src/app/mod.rs` | Replace `spinner_style: SpinnerStyle` + `pet_style: PetStyle` fields with `companion: CompanionKind`; fix `terminal_title()` to use `self.spinner_tick` |
| `src/app/mod.rs` AppState::new() | Init `companion: CompanionKind::default()` |
| `src/commands/registry.rs` | Remove `cmd("pets",…)` + `alias("emoji",…)`; add `cmd("emoji",…)`; remove `"pets"` from all_command_names; remove `("emoji","pets")` from aliases test |
| `src/commands/dispatch.rs` | Rename `"pets"` arm to `"emoji"`; rewrite `emoji_picker_items` to include spinner rows; rewrite `apply_emoji_choice` for ids 100-102 |
| `src/components/cockpit/footer.rs` | Change `app.spinner_style.glyph(tick)` to `app.companion.spinner_lead(elapsed, tick)` |
| `src/components/cockpit/footer.rs` (tests) | Update `busy_spinner_rows_at` to use `app.companion` instead of `app.pet_style` |

VERDICT: REPRODUCED — terminal_title hardcodes frame=0 (never animates); /pets is a separate command that /emoji merely aliases, and neither connects bear/cat to the spinner lead glyph.
