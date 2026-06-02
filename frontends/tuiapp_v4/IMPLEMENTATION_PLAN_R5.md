# tui_v4 — Round-5 Implementation Plan

> Specs: `recon/round5/R1..R6.md` (file:line-level, render-verified on the LIVE/styled path).
> Gate per slice = `cargo build` + `cargo test` (0 warnings) + the slice's honest-check (LIVE path) + `VERDICT: PASS`.
> **The round-3 lesson stands:** a passing unit test on a clean fixture is NOT acceptance — every honest-check must feed text through `apply_bridge_event` / render a **styled** `TestBackend` frame / drive real events.
> Baseline at kickoff: **356 passed, 0 failed** (R3 reported).

## Locked decisions (from the user + recon)

1. **Mouse = TOGGLE model (user picked Option C).**
   - **Default = NATIVE mode:** `DisableMouseCapture` + `EnableAlternateScroll` (`\x1b[?1007h`, copy Codex `tui/src/tui.rs:173-204`). The terminal does native drag-select + copy (the user's #1 pain); the wheel scrolls because the terminal translates it to arrow keys. This is exactly Codex's model.
   - **Toggle → INTERACTIVE mode** (`/mouse` and `Ctrl+Shift+M`): `EnableMouseCapture` ON → click `▸/▾` to expand/collapse any step + wheel `ScrollUp/Down`. Native selection is suppressed in this mode (switch back to native to select).
   - Make the current mode **discoverable** (the old toggle was invisible): show it in the session-info row, e.g. `mouse: select` / `mouse: click`, and/or a startup tip.
   - **Skip** an in-app drag-select highlighter — native mode already gives native selection; the toggle is the user-chosen answer. (Keeps S1 small + low-risk.)
   - Still implement Fixes **B** (▾ header for expanded turns, `FoldSegment::Text` carries `turn`) and **C** (full-width hit-zone for `NodeId::Tool`) so click-fold actually works in interactive mode.

2. **Markdown SoftBreak→hardline is SCOPED TO BLOCKQUOTES** (`quote_depth > 0`) so ordinary prose wrapping and the 4 parity invariants are untouched. (R2 FIX-C warned a global change reflows all prose.)

3. **Block-math `\,` fix = R2 Approach 4:** pre-scan the RAW source for whole-paragraph `$$…$$` blocks BEFORE pulldown eats the backslash escapes, render them directly, splice back. Inline `$…\,…$` corruption stays a documented edge.

4. **ctx% denominator (R4 D):** the implementer MUST first print the real `context_win` value/unit; if it is chars, drop the `* 3` in `cost_tracker.py`. Do not guess.

5. **Tests that encode the OLD requirement MUST be rewritten** (not deleted-around): footer.rs spinner `!contains('↑')` ×2, footer.rs `⎿`→`└` ×2, dispatch.rs `pets_picker_*`, registry `("emoji","pets")` alias test, any fold test asserting only `▸`.

6. **No GA-core edits** — only `frontends/tuiapp_v4/scripts/ga_bridge.py` and `frontends/cost_tracker.py` are editable on the python side. Keep non-test god-file budgets sane. **No commit** this round until the user authorizes.

## Slices (DO IN ORDER; halt-on-break)

- **S1 — Mouse toggle model + click expand/collapse `▸/▾` (R1).**
  Files: `src/term.rs` (default capture OFF + add `EnableAlternateScroll`/`DisableAlternateScroll`; the `set_mouse_capture`/mode toggle flips native↔interactive), `src/app/session.rs`+`src/app/types.rs` (default `mouse_capture=false`; a `MouseMode`/bool), `src/render/fold.rs` (`FoldSegment::Text { body, turn: Option<u32> }`; propagate in `fold_turns_with`), `src/markdown/mod.rs` (emit a `" ▾ title"` header tagged `NodeId::Turn` for expanded non-preamble turns; rotate `▸`↔`▾`), `src/app/fold_hit.rs` (`transcript_node_at`: full-width zone for `NodeId::Tool`, 2-cell for `NodeId::Turn`), `src/render/chip.rs` (clearer `▸ N more` / `▾ collapse` affordance), `src/components/cockpit/footer.rs` (mode indicator in session-info), `src/input/keymap.rs` + `dispatch.rs` (`/mouse` + Ctrl+Shift+M flip mode + notice).
  Honest-checks: `expanded_turn_has_downward_triangle_header_and_can_be_recollapsed`, `expanded_tool_box_affordance_row_is_clickable_at_interior_col` (R1 §5). Plus a `term`-level assert that default mode is native (capture off) and the toggle flips it.

- **S2 — Blank gap: bottom-anchor the transcript (R3).**
  File: `src/components/cockpit/transcript.rs` `render_transcript` — when `app.following() && total_visual_lines < area.height`, render the `Paragraph` into a sub-rect at `y = area.y + (h - total)` so content sits flush above the spinner.
  Honest-check: `blank_gap_bottom_anchor_live_path` (R3 §5) — gap between last content row and spinner == 0; tool box `╰─╯` present.

- **S3 — Markdown: table / headings / blockquote / HR / LaTeX `\,` (R2).**
  File: `src/markdown/render.rs` (FIX-A remove double-push in `TagEnd::TableHead`+`TableRow`; FIX-B `#` prefix + per-level modifiers H1 bold+underline … H4-6 italic; FIX-C SoftBreak→flush_line **only when `quote_depth>0`**; FIX-D width-aware HR; FIX-E block-math raw pre-scan; FIX-F no `started=true` on nested `TagEnd::List`), `src/markdown/code.rs` (`heading_style` per-level). 
  Honest-checks: `table_no_phantom_column`, `heading_levels_are_visually_distinct`, `blockquote_two_lines_render_as_two_rows`, `block_math_thin_space_not_corrupted`, `nested_list_no_spurious_blank` (R2 §5). Re-run the 4 parity invariants — they MUST stay green.

- **S4 — Spinner `↑/↓` (eased) + `└` tip + ctx/cost (R4).**
  Files: `src/components/cockpit/footer.rs` (`render_spinner`: `↑ <in> · ↓ <out>` from `tok_in/tok_out`, fall back to `↓ total`; `render_tips`: `⎿`→`└`; width-gate drops tokens first), `src/app/mod.rs` (`display_tok_in/out: Option<u64>` + ease toward target in `tick()`), `src/app/reducer.rs` (reset eased values on message-begin), `frontends/cost_tracker.py` / `scripts/ga_bridge.py` (ctx% denominator — verify unit first), `src/components/cockpit/mod.rs` (doc comment `⎿`→`└`). **Rewrite the 4-6 stale tests** (R4 §4.5).
  Honest-checks: `spinner_shows_both_up_and_down_arrows`, `tip_rows_use_floor_corner_glyph`, `display_tok_eases_toward_target_on_tick`, width-gating (R4 §5).

- **S5 — Unify `/emoji` (braille/bear/cat pick-one) + animate tab + remove `/pets` (R5).**
  Files: `src/flavor/mod.rs` (`CompanionKind { Spinner(SpinnerStyle), Pet(PetStyle) }` + `spinner_lead`/`pet_style`/`spinner_style`/`display_name`), `src/app/mod.rs` (replace `spinner_style`+`pet_style` fields with `companion`; fix `terminal_title` to use `self.spinner_tick / PET_TICKS_PER_FRAME` so the tab animates), `src/commands/registry.rs` (`/emoji` primary, delete `/pets` + the alias + names list + alias test), `src/commands/dispatch.rs` (rename arm; `emoji_picker_items` = 3 spinner + 5 pet + off = 9 rows; `apply_emoji_choice` handles ids 100-102 & 0-5), `src/components/cockpit/footer.rs` (`render_spinner` lead = `app.companion.spinner_lead(elapsed, tick)`), migrate `footer.rs` test helpers. **Delete+replace** `pets_picker_*` test.
  Honest-check: `emoji_unified_picker_and_animated_title` (R5 §5) — `/pets` gone, picker has 9 rows, bear drives the spinner lead, `terminal_title` differs at tick 0 vs 5.

- **S6 — Sticky last-user-message header on scroll (R6 Part A).**
  Files: `src/app/mod.rs` (`last_user_source_first_line`), `src/components/cockpit/mod.rs` (`sticky_header: Option<Rect>` slot in `split_cockpit`, inserted as `Length(1)` only when `!following && has last user`), `src/components/cockpit/transcript.rs` (`render_sticky_header` → dim UserBand `↑ <prompt…>`), call it in `render_cockpit`.
  Honest-check: `sticky_header_shows_when_scrolled_up_absent_when_following` (R6 §A5).

- **S7 — `/keybindings` overlay doc clarity (R6 Part B GAP-1..4).**
  File: `src/components/overlay/info.rs` — ensure the overlay clearly lists the intentional v3→v4 remaps: `Ctrl+Shift+O = fold all` (and `Ctrl+O = copy last reply`), `Ctrl+G = stash draft`, `Ctrl+B = branch`, `Ctrl+S = session dashboard`, and the new `/mouse` mode toggle. No behavior change — documentation only.

## Monitor (final adversarial gate)
Render-based verifiers (one per slice cluster) citing rendered-frame evidence, NOT "tests green". Then: full `cargo test` (0 warnings), 4 parity invariants green, rebuild release exe, update `checklist.md` + `query.md` + memory.
