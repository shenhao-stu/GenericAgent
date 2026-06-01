# C4 — Theme redesign + per-command effects + char effects

Scope: `/theme` (rename `ga-default`→`default`, redesign dark+light sets) and `/effects`
(per-command border + char identity for `/goal /hive /conductor /morphling`).
Authoritative reference: CC `temp/claude-code/src/utils/theme.ts` (the real token tables).
Peer: `openai/codex` `codex-rs/tui` (`shimmer.rs`, `motion.rs`, `frames.rs`, `theme_picker.rs`,
`style.rs`/`color.rs`) per `recon/codex_tui_patterns.md` §4/§8/§8a.

---

## Findings (file:line bugs, root cause not symptom)

### F1 — `ga-default` naming is wired in 9 places; rename is not a one-liner (Q11 rename ask)
Root cause: the theme identity string `"ga-default"` is duplicated as a literal across the
constructor, the registry, three tests, and is *load-bearing* in the picker (it round-trips
through `app.theme.name` ↔ `by_name` ↔ `index_of`). All must move together or the picker
seed / commit-revert breaks.
- `theme/mod.rs:94` `Default::default → Theme::ga_default()`
- `theme/mod.rs:131-133` `pub fn ga_default()` + `name: "ga-default"`
- `theme/mod.rs:328` registry `("ga-default", Theme::ga_default)` — **first entry; picker order index 0**
- `theme/mod.rs:388,411,431,435,459` tests assert the literal `"ga-default"` and `index_of("ga-default")==Some(0)`
- `main.rs:1657` `PickItem::new(i,*n).current(*n == app.theme.name)` — compares against the name
- `main.rs:1674` `theme::index_of(app.theme.name)` seeds picker selection
CC precedent: `theme.ts:91-98` names are bare semantic strings (`'dark'`,`'light'`,…) with NO
product prefix; default is `dark` (`getTheme` default branch `:610`). So the correct target is
`"default"` (not `"ga-default"`), matching CC's convention exactly.

### F2 — tui_v4's per-theme rainbows diverge from CC's single canonical ROYGBIV (palette quality)
Root cause: every theme defines its OWN 7-stop rainbow (`theme/mod.rs:150,182,214,246,278,311`)
in saturated web-RGB (`0xff5c57…`). CC ships ONE muted ROYGBIV reused verbatim in *all six*
themes (`theme.ts:177-183` etc.): `red rgb(235,95,87) · orange (245,139,87) · yellow (250,195,95)
· green (145,200,130) · blue (130,170,220) · indigo (155,130,200) · violet (200,130,180)`.
The tui_v4 rainbows are louder than CC's and inconsistent stop-to-stop across themes, which is
exactly the "not Claude-Code-ish" look the recon flagged (`effects_engine.md` §4.6: "match the 7
stops for fidelity"). Symptom you'd see: the header separator and (new) command borders look
neon, not the soft CC gradient.

### F3 — composer border FX is command-AGNOSTIC; the whole C4 premise is unimplemented (Q11 core)
`components/effects_paint.rs:189 draw_composer_border_fx` paints the SAME flowing rainbow +
SAME dingbat set (`['✦','✧','·','✶','∗']`, line 257) regardless of which command is typed.
Root cause: the gate `components/mod.rs:157 if fx_command_active(...)` collapses the command
to a `bool` (`mod.rs:879-885 fn fx_command_active`) — the *identity* of the command is thrown
away before the painter runs. So `/goal`, `/hive`, `/conductor`, `/morphling` are visually
identical. The deliverable (b) — a DISTINCT effect per command — cannot exist until the painter
receives *which* command.

### F4 — the typed command text (`/goal`) is rendered PLAIN; char-effect ask unimplemented (Q11)
`components/mod.rs:505 text_style = …Token::Text` then `:543/:548/:550` push the row spans with
that flat style. Root cause: `composer_lines` (`mod.rs:494`) has no notion of "the leading
command word is special" — it styles the entire buffer as body text. The ask ("`/goal` 这些字符
本身也要做特效渲染") has no implementation hook at all. Note the existing precedent right next to
it: shell mode ALREADY special-cases the leading `!` sigil in hot-pink (`mod.rs:523-535
bang_pink`) — the same seam is where a command-word style belongs.

### F5 — border FX ignores `prefer_fg_only` (tmux truecolor-bg gap) — pre-existing, inherited by C4
`draw_composer_border_fx` gates only on `caps.enabled()` (`effects_paint.rs:190`) and always
sets `fg` (fine), but the surrounding engine doc (`effects/mod.rs:16` tenet 3) and `caps.prefer_fg_only()`
(`effects/mod.rs:150`) exist precisely so effects degrade under tmux. The border is fg-only so it
*happens* to be safe, but any new per-command effect that wants a bg tint (e.g. conductor's
"sweep" idea) MUST consult `prefer_fg_only()` or it will silently break under tmux — call this
out so the new effects don't regress the lesson CC paid for (`theme.ts:617-620` Apple-Terminal
256 downshift; recon §4.6 "3 hostile terminals").

### F6 — `Token` set has no neutral "accent-2"/inverse-text token for light themes (palette redesign blocker)
`Token::ALL` (`theme/mod.rs:61-76`) has 14 tokens; there is NO `InverseText` (text-on-accent) and
NO second accent. CC carries `inverseText` (`theme.ts:129` white / `:535` black) specifically so a
filled pill/band stays legible in BOTH light and dark. tui_v4's `light()` (`theme/mod.rs:292`) has
`UserBand rgb(0xe8e8e8)` (light) but `Text` is near-black `rgb(0x1a1a1a)` — fine — yet any future
*filled* accent chip on light ground has no contrasting text token. Minor for C4 (we can lean on
`Text`/`UserBand`), but flag it: the redesign should add `InverseText` if filled command badges
are wanted.

---

## Competitor patterns (CC / Codex / v2 / v3, with file cites)

### CC `utils/theme.ts` — THE reference (read in full)
- **Naming** `theme.ts:91-98`: `THEME_NAMES = ['dark','light','light-daltonized','dark-daltonized','light-ansi','dark-ansi']`. Default resolves to `darkTheme` (`getTheme` `:598-612` default arm). → confirms `ga-default`→`default` and that the *primary* theme is a DARK one.
- **Dark tokens** (`darkTheme` `:440-515`) — tui_v4's `ga_default` already copies these (claude `rgb(215,119,87)`, success `(78,186,101)`, error `(255,107,128)`, warning `(255,193,7)`, suggestion `(177,185,249)`, subtle `(80,80,80)`, bashBorder `(253,93,177)`, ide `(71,130,200)`, userMessageBackground `(55,55,55)` — note CC moved this to **55** not the 58 tui_v4 uses, and `claudeShimmer (235,159,127)`). Use these verbatim for `default`.
- **Light tokens** (`lightTheme` `:115-191`) — the real light-ground values tui_v4's `light()` only approximates: `text rgb(0,0,0)`, `subtle rgb(175,175,175)` (NOT 0x70 grey — too dark on white), `success rgb(44,122,57)`, `error rgb(171,43,63)`, `warning rgb(150,108,30)`, `suggestion/permission rgb(87,105,247)`, `bashBorder rgb(255,0,135)`, `claude rgb(215,119,87)` (same brand), `userMessageBackground rgb(240,240,240)`, `promptBorder rgb(153,153,153)`. **Lesson:** CC keeps the brand orange identical light/dark and only swaps neutrals + darkens semantics for contrast on white.
- **Shimmer convention** `theme.ts:119,444` etc.: every `*Shimmer` = a *lighter* step of its base (claude→claudeShimmer = +30 on each channel-ish). The recon §4.6 codifies this; tui_v4 already follows it (`ClaudeShimmer` token + `EffectPalette.shimmer_glow`). Reuse for per-command char shimmer.
- **Canonical rainbow** `theme.ts:177-190` (and identical in all 6): the muted ROYGBIV + its `_shimmer` pair. This is the per-command effect color *source*.

### Codex `codex-rs/tui` — the direct Rust+ratatui peer (recon `codex_tui_patterns.md`)
- **§4 `shimmer.rs`**: the raised-cosine sheen `0.5*(1+cos(π·d/half))` on a fixed 2.0s period; highlight = `blend(default_bg, default_fg, t*0.9)`. **tui_v4's `effects/shimmer.rs:75 intensity_at` is the SAME formula** — peer-validated. Codex keys the blend off the *terminal* fg/bg; tui_v4 keys off theme tokens (`shimmer_base→shimmer_glow`). Adopt Codex's per-char sweep math for the command-word char effect (it's already in the crate).
- **§4 `frames.rs` + `theme_picker.rs`**: Codex embeds **10 named spinner frame-sets** (`default/codex/openai/blocks/dots/hash/hbars/vbars/shapes/slug`, 36 frames each, 80ms/frame) and `theme_picker.rs` selects the **spinner VARIANT per context** — i.e. *the animation identity is a selectable named preset*, not a hardcoded loop. This is the exact precedent for "a distinct effect per command": model each command's FX as a named preset chosen by a selector, not five copy-pasted painters.
- **§4 `motion.rs` accessibility guardrail**: a unit test enforces that `spinner()`/`shimmer_spans()` are ONLY called via the motion abstraction (`MotionMode::{Animated,Reduced}`). → the per-command FX dispatch should likewise funnel through ONE entry (so reduced-motion / caps gating lives in one place, not per command).
- **§8 terminal-adaptive theming** (`terminal_palette.rs`/`style.rs`/`color.rs`): Codex derives semantics from the REAL terminal fg/bg (OSC 10/11 probe), and **`disallowed_methods` lint FORBIDS raw `Color::Rgb`** — every color flows through `best_color()`. This is the strongest possible version of the review-principle "no hardcoded RGB at call sites"; tui_v4 already centralizes via `Token`, but the per-command palettes MUST stay inside `theme/`/`EffectPalette`, never inline in `effects_paint.rs`.
- **§8a `Renderable`**: height-first compose contract — not needed for C4 but confirms Codex keeps render logic data-driven + unit-tested (snapshot tests, §9.10). The new per-command FX should get a pure unit test like the existing `fx_command_active_only_for_orchestration` (`mod.rs:989`).

### v3 (`tui_v3.py`, recon `tui_v3_inventory.md` §7)
- Single fixed palette, ONE accent `_ACCENT = #5e6ad2` (`tui_v3.py:1503`); shell pink `_SHELL_ACCENT 38;5;205` (`:1518`). No multi-theme, no per-command FX.
- **Light-mode lesson** `tui_v3.py:1475-1488 _MD_THEME`: "NO dim on body (dim-on-white unreadable)". → directly supports F2/F6: the `light` theme's `Dim` must be a *real* light grey (CC's `rgb(175,175,175)`), and effects that rely on dimming must not wash out on white.
- Apple-Terminal 256-color pin (`:1499-1508`) — same hostile-terminal lesson as CC/Codex.

### v2 (`tuiapp_v2.py`) — n/a for theming (no token system; superseded by v3 then v4).

### Well-regarded palettes for the redesign (cited hex)
- **Catppuccin Latte** (light) — base `#eff1f5`, text `#4c4f69`, subtext `#6c6f85`, mauve `#8839ef`, green `#40a02b`, red `#d20f39`, yellow `#df8e1d`, blue `#1e66f5`, teal `#179299`, pink `#ea76cb`. Source: catppuccin.com/palette. A softer, more modern light theme than CC's stark white-ground `lightTheme`.
- **Solarized Light** (light) — base3 `#fdf6e3` (bg), base00 `#657b83` (body), base01 `#586e75` (emph), yellow `#b58900`, red `#dc322f`, green `#859900`, blue `#268bd2`, violet `#6c71c4`, magenta `#d33682`, cyan `#2aa198`. Source: ethanschoonover.com/solarized. Canonical, low-contrast easy-on-eyes.
- **Tokyo Night / Nord / Gruvbox / Dracula** (dark) — already in tui_v4 (`theme/mod.rs:163-321`), keep; they are the de-facto dark set and Codex bundles the same syntect themes (recon §7: "catppuccin/dracula/gruvbox/nord/solarized").

---

## Fix design (Rust sketches: the actual changed lines / new fn signatures)

### (a) Rename + redesigned theme set

**Rename `ga-default`→`default`** — change the literal in all 9 sites. The constructor name stays
`ga_default` (or rename to `default_theme` to avoid clashing with the `Default` trait):

```rust
// theme/mod.rs:131-133
pub fn default_theme() -> Self {                 // was ga_default
    Theme { name: "default", palette: [ /* CC darkTheme verbatim */ ], rainbow: CC_RAINBOW }
}
// theme/mod.rs:94
impl Default for Theme { fn default() -> Self { Theme::default_theme() } }
// theme/mod.rs:328 registry — keep FIRST so picker index 0 == default
pub const THEME_BUILDERS: &[(&str, fn() -> Theme)] = &[
    ("default",     Theme::default_theme),
    ("tokyo-night", Theme::tokyo_night),   // promote a dark as the 2nd
    ("nord", Theme::nord), ("gruvbox", Theme::gruvbox), ("dracula", Theme::dracula),
    ("catppuccin-latte", Theme::catppuccin_latte),  // NEW light
    ("solarized-light",  Theme::solarized_light),   // NEW light
    ("light", Theme::light),                        // keep CC stark-white light
];
```
Tests at `:388,411,431,459` flip `"ga-default"`→`"default"` and the `index_of` expectations.
`main.rs:1657/1674` need NO change (they read `app.theme.name` dynamically) — that is why the
literal must be consistent everywhere.

**Adopt CC's canonical rainbow as a shared const** (fixes F2):
```rust
// theme/mod.rs — one source of truth, reused by every theme's `rainbow:` field
const CC_RAINBOW: [Color; 7] = [
    Color::Rgb(235,95,87), Color::Rgb(245,139,87), Color::Rgb(250,195,95),
    Color::Rgb(145,200,130), Color::Rgb(130,170,220), Color::Rgb(155,130,200),
    Color::Rgb(200,130,180),
];
```
(Dark themes with their own identity — dracula/gruvbox — MAY keep a themed rainbow; but
`default`, `light`, and the two new light themes should use `CC_RAINBOW` so the separator reads
CC-soft. This collapses 6 ad-hoc rainbow arrays toward 1 — review principle #12.)

**New light themes** (token order = `Token::ALL`):
```rust
pub fn catppuccin_latte() -> Self { Theme { name: "catppuccin-latte", palette: [
    Color::Rgb(0x88,0x39,0xef), // Claude→mauve (brand-accent on light)
    Color::Rgb(0x4c,0x4f,0x69), // Text  #4c4f69
    Color::Rgb(0x8c,0x8f,0xa1), // Dim   subtext0 (real light grey, not too dark — F6/v3 lesson)
    Color::Rgb(0x40,0xa0,0x2b), // Success green
    Color::Rgb(0xdf,0x8e,0x1d), // Warning yellow
    Color::Rgb(0xd2,0x0f,0x39), // Error red
    Color::Rgb(0x1e,0x66,0xf5), // Suggestion blue
    Color::Rgb(0x88,0x39,0xef), // PlanMode mauve
    Color::Rgb(0xfe,0x64,0x0b), // AutoAccept peach
    Color::Rgb(0xbc,0xc0,0xcc), // Border  surface1
    Color::Rgb(0xe6,0xe9,0xef), // UserBand mantle (light band)
    Color::Rgb(0xea,0x76,0xcb), // ShellAccent pink
    Color::Rgb(0x1e,0x66,0xf5), // Ide blue
    Color::Rgb(0x7287,0xfd & 0,0)/*see note*/ ], rainbow: CC_RAINBOW } }  // ClaudeShimmer = lighten(mauve)
```
(Use a `lighten(base, k)` helper for every `*Shimmer`/glow rather than hand-tuned hex — see below.)
`solarized_light()` analogous with base3 `#fdf6e3` ground assumption: Text `#657b83`, Dim `#93a1a1`,
Success `#859900`, Error `#dc322f`, Warning `#b58900`, Suggestion `#268bd2`, Border `#eee8d5`,
UserBand `#eee8d5`, ShellAccent `#d33682`, Ide `#268bd2`.

**Helper to kill hand-tuned shimmer hex (review #9/#12):**
```rust
/// A lighter step of `c` toward white by `k∈[0,1]` (the CC `*_shimmer` convention).
fn lighten(c: Color, k: f32) -> Color { crate::effects::shimmer::blend(c, Color::Rgb(255,255,255), k) }
```
so `ClaudeShimmer = lighten(claude, 0.18)` — one rule, every theme, instead of 6 magic RGBs.

### (b) Distinct effect identity per command

Introduce a `FxCommand` enum next to `fx_command_active` and have the painter switch on it.
Each command gets: a **border palette** (which colors flow on the box) + a **border glyph/motion**
+ a **char-effect** for the typed word. Rationale per command keyed to its nature:

| cmd | identity | border-effect | char-effect (typed word) | rationale |
|---|---|---|---|---|
| `/goal` | a single guiding star / aim | **steady pulse** (breathing, NOT flowing): whole border = `Token::Claude`, brightness `ping_pong` via `shimmer.phase`; corner glyph `◆`. Particles: a few `✦` that *converge* toward top-center. | `/goal` shimmers claude→claudeShimmer (sheen sweeps once L→R, then holds) | a goal is fixed & focal — a calm pulse, not chaos; brand-orange = the "north star". |
| `/hive` | swarm / many agents | **traveling dots** around the perimeter (the existing flow, but mono-accent `Token::Success` mint, not rainbow); glyph `⬡` (hex = honeycomb). Particles: 6 `·`/`∘` orbiting (more in full). | `/hive` = each char a slightly different mint shade (a "swarm" of hues around `Success`) | bees/cells → hexagon + busy moving motes; green = "live workers". |
| `/conductor` | orchestration / baton sweep | **L→R sweep**: a bright band (raised-cosine) travels the TOP edge only, rest dim `Token::Suggestion` blue; glyph `▸`. Like a conductor's baton beat. | `/conductor` = a left-to-right highlight sweep (sheen) on the word, looping (the "downbeat") | a conductor sets tempo → one directional sweep, metronomic; blue = calm command. |
| `/morphling` | shape-shift / transform | **rainbow morph**: KEEP the full ROYGBIV flow (the only command that earns the rainbow) but slow + glyph cycles `◆◇○◌` per corner (shape-shifting). Particles cycle glyph set `['✦','✶','❄','∗']`. | `/morphling` = hue-cycling per char (each char marches through `CC_RAINBOW` over time) | morph = identity in flux → the multi-color, multi-glyph one; reuses today's rainbow so we don't lose it. |

```rust
// components/mod.rs — replace the bool with an identity (keep the old fn as a thin wrapper)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FxCommand { Goal, Hive, Conductor, Morphling }

pub fn fx_command(text: &str) -> Option<FxCommand> {
    let rest = text.trim_start().strip_prefix('/')?;
    let word = rest.split(char::is_whitespace).next().unwrap_or("");
    Some(match word {
        "goal" => FxCommand::Goal, "hive" => FxCommand::Hive,
        "conductor" => FxCommand::Conductor, "morphling" => FxCommand::Morphling,
        _ => return None,
    })
}
#[inline] pub fn fx_command_active(text: &str) -> bool { fx_command(text).is_some() } // keep tests green
```

```rust
// theme/mod.rs — per-command border palette derives from EXISTING tokens (no new RGB):
impl FxCommand {
    /// (flow_color source, motion) — colors come from theme tokens / CC_RAINBOW only.
    pub fn border(self, theme: &Theme) -> FxBorder {
        match self {
            FxCommand::Goal      => FxBorder::Pulse  { color: theme.color(Token::Claude),      corner: '◆' },
            FxCommand::Hive      => FxBorder::Orbit  { color: theme.color(Token::Success),     corner: '⬡' },
            FxCommand::Conductor => FxBorder::Sweep  { color: theme.color(Token::Suggestion),  corner: '▸' },
            FxCommand::Morphling => FxBorder::Rainbow{ stops: *theme.rainbow(),                corner: '◆' },
        }
    }
}
pub enum FxBorder { Pulse{color:Color,corner:char}, Orbit{color:Color,corner:char},
                    Sweep{color:Color,corner:char}, Rainbow{stops:[Color;7],corner:char} }
```

```rust
// components/effects_paint.rs:189 — take the command, switch the per-edge color fn.
pub fn draw_composer_border_fx(frame:&mut Frame, app:&AppState, area:Rect, now_ms:u64, cmd:FxCommand){
    if !app.effects.caps.enabled() || area.width<2 || area.height<2 { return; }
    if app.composer.is_shell_mode() { return; }
    let t = now_ms as f32 * 0.00007;
    // ONE perimeter-index→color closure, parameterized by the command's FxBorder:
    let color_at: Box<dyn Fn(usize)->Color> = match cmd.border(&app.theme) {
        FxBorder::Rainbow{stops,..} => Box::new(move |k| flow_color(&stops, k as f32/perim + t)),
        FxBorder::Pulse{color,..}   => { let g=lighten(color,0.30);
            let b=0.5*(1.0+(t*6.0).cos());                 // breathing, no travel
            Box::new(move |_| blend(color,g,b)) }
        FxBorder::Orbit{color,..}   => { let g=lighten(color,0.30);
            Box::new(move |k| blend(color,g, intensity_at((t).rem_euclid(1.0),0.12,k,perim as usize))) }
        FxBorder::Sweep{color,..}   => { let g=lighten(color,0.35);   // sweep only matters on top edge
            Box::new(move |k| blend(color,g, intensity_at((t).rem_euclid(1.0),0.18,k,perim as usize))) }
    };
    // …existing top/bottom/side loops, but call color_at(..) and use cmd corner glyph…
}
```
The per-edge index math (`effects_paint.rs:204-271`) is REUSED unchanged; only the color source and
corner glyph vary — so four identities cost ~one `match`, not four painters (review #5/#12).

### (b cont.) Char-effect on the typed command word

Add a tiny pure helper + call it from `composer_lines` for the FIRST row's leading `/word`:
```rust
// components/mod.rs — style each grapheme of the command word per its FxCommand.
fn command_word_spans<'a>(word:&'a str, cmd:FxCommand, theme:&Theme, phase:f32) -> Vec<Span<'a>> {
    let base = match cmd { FxCommand::Hive=>Token::Success, FxCommand::Conductor=>Token::Suggestion,
                           _=>Token::Claude };
    word.char_indices().map(|(i,ch)| {
        let n = word.chars().count().max(1);
        let col = i as f32 / n as f32;
        let fg = match cmd {
            FxCommand::Morphling => flow_color(theme.rainbow(), col + phase),       // hue march
            FxCommand::Hive      => blend(theme.color(base), lighten(theme.color(base),0.4), (col*6.0).fract()),
            _ /*Goal,Conductor*/ => { let t = intensity_at(phase.rem_euclid(1.0),0.25,i,n); // sheen sweep
                                      blend(theme.color(base), lighten(theme.color(base),0.35), t) }
        };
        Span::styled(ch.to_string(), Style::default().fg(fg).add_modifier(Modifier::BOLD))
    }).collect()
}
```
Wire at `components/mod.rs:509-551` (the `for (ri,r)` loop): when `ri==0` and `fx_command(text)`
is `Some(cmd)` and the cursor is NOT inside the word, replace the flat first-token span with
`command_word_spans(word, cmd, theme, app.effects.clock)`; the remainder of the row keeps
`text_style`. Mirror the existing `bang_pink` peel (`mod.rs:523-535`) — same seam, same
"only-on-row-0, skip if cursor sits on it" guard.

### (c) Wiring plan (composer picks the effect by detected command)

1. `components/mod.rs:157` change the call site:
   ```rust
   if let Some(cmd) = fx_command(app.composer.text()) {
       effects_paint::draw_composer_border_fx(frame, app, composer, now_ms, cmd);
   }
   ```
2. `render_composer`→`composer_lines` gains `now_ms`/`clock` so the char-effect can animate;
   pass `app.effects.clock` (already the single pure clock, `effects/mod.rs:263`). The composer is
   redrawn every tick while text is present, so the sweep/hue advance "for free" (same path the
   border already uses).
3. **Animation cadence**: the per-command border + char effect should run even in `EffectMode::Off`
   (today's gate is intentionally independent of `/effects` mode — `mod.rs:152-156` comment), but
   funnel the reduced-motion check through ONE spot: if a future `reduced_motion`/caps says static,
   `color_at` returns the flat base color and `command_word_spans` drops the phase term. (Codex
   `motion.rs` guardrail, recon §4.)
4. **Demo**: extend `/effects demo` to cycle the four command borders so the picker shows them
   (optional; `app/mod.rs:566 start_effects_demo`).
5. **Tests** (pure, mirror `mod.rs:989`): `fx_command` maps the 4 words and rejects `/hivemind`
   etc.; `command_word_spans("goal",Goal,…,0.0)` returns 4 spans all with a fg set; each
   `FxBorder` variant yields a different `color_at(0)` for a fixed `t` (proves distinctness).

---

## Review-principle violations (cite principle # + file:line)

- **#5 (linear growth) / #12 (more features fewer lines)** — `effects_paint.rs:189-271`: implementing
  4 distinct effects by *copying* `draw_composer_border_fx` four times would be the wrong fix. The
  sketch above adds 4 identities via ONE `match` over a `FxBorder` enum + reused edge loops →
  features up, lines ~flat. (If a reviewer instead sees four near-identical painter fns, that's the
  regression.)
- **#6 (constraints in types)** — `components/mod.rs:879 fx_command_active` returns `bool`, discarding
  the command identity that the painter then *cannot* recover. Replacing the predicate with
  `fx_command(&str)->Option<FxCommand>` puts the constraint ("FX only for these 4, and the painter
  needs to know which") into the type, not into a re-parse. Symptom of the current shape: F3/F4 are
  literally impossible to implement without this change.
- **#1 (module boundaries) / "no hardcoded RGB at call sites"** — `effects_paint.rs:257` hardcodes the
  dingbat set and (in any naive per-command fix) would hardcode per-command colors in the *paint*
  layer. CC/Codex both forbid this (CC `Token` table; Codex `disallowed_methods` on `Color::Rgb`,
  recon §8/§11). Per-command palettes/glyphs must live in `theme/`/`FxCommand::border`, leaving
  `effects_paint.rs` a pure renderer. Today the border color already comes from `theme.rainbow()`
  (good) — keep that discipline for the 3 new identities.
- **#9 (self-explanatory, minimal comments) / #10 (visual uniformity)** — `theme/mod.rs:131-321`: the six
  palettes carry hand-tuned `*Shimmer`/glow hex (`235,159,127`, `0xa3d4e4`, …) with per-line "lighter
  X" comments. Replacing them with `lighten(base, 0.18)` removes ~14 magic RGBs + their explanatory
  comments per theme — the code states the rule instead of restating it in prose.
- **#8 (consistency, no surprises)** — `theme/mod.rs:9` doc-comment says "ships 6 named palettes
  (`ga-default`…)"; after rename it must say `default`. Stale identity strings in doc + 3 tests are a
  consistency trap (F1). Also: `Theme::default()` calling `ga_default()` while the registry key is a
  different string is exactly the kind of "same concept, two spellings" #8 warns against.
- **#2 (local reasoning)** — POSITIVE: keep `fx_command`/`FxCommand::border`/`command_word_spans` PURE
  (no `Instant::now()`, phase passed in) so each is unit-testable in isolation, matching the existing
  `effects/` tenet (`effects/mod.rs:5-13`) and Codex's snapshot-test discipline (recon §9.10).

---

## Open questions / risks

1. **Rename migration**: does any persisted setting / session file store `theme.name == "ga-default"`?
   `main.rs` reads `app.theme.name` but if a config writer persists the string, renaming silently
   resets users to `default` on next launch. Need to grep the persistence path (not in C4 files) and
   add a `"ga-default"→"default"` alias in `by_name` (`theme/mod.rs:343`) for one release. **Cheap
   insurance: keep `by_name("ga-default")` resolving to `default`.**
2. **`/morphling` keeps the rainbow** — is that acceptable given Q11 wants *distinct* identities? Rationale:
   morph = flux, so rainbow is its natural fit and we avoid throwing away the existing flow code. If the
   reviewer wants all four non-rainbow, conductor/goal/hive already are; only morphling overlaps the
   old always-on look. Confirm intent.
2b. **Two new LIGHT themes vs. ask "1-2 light"** — delivering Catppuccin-Latte + Solarized-Light + the
   existing stark CC `light` = 3 light. Drop CC `light` if only 2 wanted; but it's the
   highest-contrast option (accessibility), so I'd keep all 3.
3. **Char-effect + cursor interaction**: when the cursor sits inside `/goal` (mid-typing), the
   inverse-cursor cell (`mod.rs:544-547`) must win over the char shimmer, else the cursor vanishes into
   the sheen. The sketch guards "skip if cursor in word" but the exact column math (peeled-`!` shift,
   `mod.rs:540`) needs care for the `/`-prefix. Risk of an off-by-one cursor glitch.
4. **Performance**: char-effect recolors per grapheme every tick while a command is typed. Bounded
   (command word ≤ ~10 chars, ~10fps) so negligible, but `Box<dyn Fn>` per frame in `color_at`
   allocates — prefer a non-boxed `enum`-dispatch or compute the 4 cases inline to honor the
   "no per-frame alloc" tenet (`effects/mod.rs:6`). Minor.
5. **tmux/Apple-Terminal** (F5): the new Sweep effect is fg-only (safe), but if anyone adds a bg tint
   for conductor's band, it MUST consult `caps.prefer_fg_only()` (`effects/mod.rs:150`) — call out in
   review so the hostile-terminal lesson isn't relitigated.
6. **Rainbow consolidation** (F2): switching `default`/light themes to `CC_RAINBOW` changes the
   header-separator look for existing users. It's the *intended* CC-fidelity fix, but it's a visible
   change — flag for sign-off. Dracula/Gruvbox can keep themed rainbows (they're identity themes).
