# tui_v4 — Round-5 checklist (LIVE tracker)

> Verification rule (unchanged from R4): a box ticks ONLY on evidence from the
> **live/styled** path — text fed through `apply_bridge_event`, the styled
> `TestBackend` cell grid scanned, or real key/mouse events. A passing unit test on a
> hand-built fixture is NOT acceptance. Specs: `recon/round5/R1..R6`. Plan:
> `IMPLEMENTATION_PLAN_R5.md`.
> **Result: v0.5.0 — 370 passed / 0 failed / 0 warnings; 4 parity invariants green;
> release exe built (9.6s); adversarial Monitor (5 Opus verifiers) ALL CONFIRMED.**

---

## ⚠️ The round-5 incident (write it down)
A **Sonnet** implementation agent (S3) shipped an **infinite loop** in the new
`extract_block_math_paragraphs` (`render.rs`): at end-of-input (`i == len`) it did
`continue` without advancing `i`, spinning the empty trailing chunk forever. Because
that helper runs on EVERY markdown render, the slice's `cargo test` gate hung; the
hung test binaries (`tui_v4-<hash>.exe`) pinned the CPU and **froze the user's
machine**. Fix: `break` at end-of-source (`render.rs:125`). Hardening adopted:
(1) implementation/finish work moved to **Opus** (user directive — strongest model);
(2) every later gate runs `timeout … cargo test` so a hang self-kills, never spins;
(3) one `cargo` at a time. The Monitor compiled a standalone replica of the loop and
proved it terminates on empty/EOF/no-newline inputs.

## Slices (all DONE, gate = build + test 0/0 + live honest-check + Monitor)

- [x] **S1 — mouse TOGGLE model + click expand/collapse ▸/▾.** Default = NATIVE
  (`term.rs`: no `EnableMouseCapture`, `EnableAlternateScroll`=`?1007h` → native OS
  drag-select + wheel-as-arrows, the Codex model); `/mouse` + `Ctrl+Shift+M` toggle
  capture ON for click-fold/wheel; footer shows `mouse: select`/`mouse: click`.
  `FoldSegment::Text{turn}` + a ` ▾ <title>` header tagged `NodeId::Turn` for expanded
  turns (re-collapsible); full-width hit-zone for `NodeId::Tool`. Same `NodeId::Turn`
  drives ` ▸ `(folded) ⇄ ` ▾ `(expanded). *Tests:* `expanded_turn_has_downward_triangle_header_and_can_be_recollapsed`, `expanded_tool_box_affordance_row_is_clickable_at_interior_col`. **Monitor M1 CONFIRMED.**
- [x] **S2 — blank-gap bottom-anchor.** `transcript.rs render_transcript` renders into
  a sub-rect at `y+gap` when `following() && total < height` so content sits flush above
  the spinner; scrolled-up/overflow paths provably un-shoved. *Test:* `blank_gap_bottom_anchor_live_path` (gap==0, `╰─╯` present). **Monitor M3 CONFIRMED.**
- [x] **S3 — markdown.** Table phantom-column removed (TableHead/TableRow double-push);
  H1–H6 distinct by **modifier+color, NO bare `#`** (round-4 rule kept — FIX-B's hash
  prefix reverted); blockquote SoftBreak→newline (scoped to `quote_depth>0`); width-aware
  HR; **LaTeX `\,` pre-scan** (`extract_block_math_paragraphs`, the loop now `break`s at
  EOF); nested-list spurious blank fixed (`TagEnd::Item` flushes only with content).
  *Tests:* `table_no_phantom_column`, `heading_levels_are_visually_distinct`, `blockquote_two_lines_render_as_two_rows`, `block_math_thin_space_not_corrupted`, `nested_list_no_spurious_blank` + the two styled-frame heading tests. **Monitor M2 CONFIRMED.**
- [x] **S4 — spinner `↑in · ↓out` (eased) + `└` tip.** `render_spinner` shows BOTH
  arrows from `display_tok_in/out`, eased toward `tok_in/tok_out` in `tick()` (single
  bounded step, NOT a loop); `⎿`→`└` in `render_tips`. ctx% LEFT truthful (the
  `context_win*3` denominator matches GA's real trim trigger — NOT a bug; not touched).
  Stale ↓-only / `⎿` tests rewritten. *Tests:* `spinner_shows_both_up_and_down_arrows`, `tip_rows_use_floor_corner_glyph`, `display_tok_eases_toward_target_on_tick`. **Monitor M4 CONFIRMED.**
- [x] **S5 — unified `/emoji` (braille/bear/cat pick-one) + animated tab + `/pets` gone.**
  `CompanionKind { Spinner(SpinnerStyle), Pet(PetStyle) }` replaces the two fields; one
  9-row picker (3 spinner ids 100-102 + 5 pet + Off); selection drives the spinner LEAD
  glyph AND the tab; `terminal_title` animates (`spinner_tick / PET_TICKS_PER_FRAME`);
  `/pets` removed (was a separate cmd, `/emoji` now primary). *Test:* `emoji_unified_picker_and_animated_title`. **Monitor M4 CONFIRMED.** *(The half-applied migration from the stopped Sonnet run was finished on Opus.)*
- [x] **S6 — sticky last-user-prompt header on scroll.** `last_user_source_first_line()`
  + a `sticky_header: Option<Rect>` slot (`Length(1)` above the transcript, only when
  `!following && a prompt exists`) → dim `UserBand` `↑ <prompt…>`. *Test:* `sticky_header_shows_when_scrolled_up_absent_when_following`. **Monitor M5 CONFIRMED.**
- [x] **S7 — `/keybindings` doc clarity.** Added `Ctrl+G` (stash, moved from `Ctrl+S`) +
  confirmed the v3→v4 remaps (`Ctrl+Shift+O` fold, `Ctrl+O` copy, `Ctrl+S` dashboard,
  `Ctrl+B` branch, `Ctrl+Shift+M`/`/mouse` mode + native-select hint) are all listed.
  *Test:* `keybindings_overlay_lists_v3_to_v4_remaps`. **Monitor M1 CONFIRMED.**

## Monitor (adversarial, render-based, 5 Opus verifiers)
- [x] **M1 mouse+fold · M2 markdown · M3 blank-gap · M4 spinner+emoji · M5 sticky+CPU-safety — ALL CONFIRMED, zero gaps.** M5 (the user's top concern) proved loop safety: suite finishes **1.39s**, every round-5 `while` provably advances/breaks (EOF-`break` at render.rs:125 + a compiled replica hammered on empty/EOF inputs), and **0** `tui_v4`/`cargo`/`rustc` processes leak.

## Pending (user's call)
- [ ] Commit / push / release v0.5.0 — NOT done (no authorization this round; awaiting user).
- [ ] Live aesthetic confirmation (native select+copy feel in the user's terminal; per-command FX colors).
