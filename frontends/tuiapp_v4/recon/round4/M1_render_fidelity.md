# M1 Render Fidelity — Round-4 Adversarial Verification Report

**Method:** PYTHONUTF8=1 cargo run --dump-frame <scenario> + code reading + unit test execution + reference screenshot D:/Screenshots/clip_20260601_134922.png comparison. All dump-frame output is at 100x30.

---

## CLAIM 1 — HEADER is a multi-line ROUNDED box

**Verdict: CONFIRMED**

Rendered evidence from `--dump-frame normal`:
```
╭──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ >_ GenericAgent                                                                                  │
│                                                                                                  │
│ model:   codex-pro · gpt-5.5   /llm switch                                                       │
│ directory:   D:\GenericAgent                                                                     │
│ session:   session 1 · scrollback                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

- Rounded corners (╭/╰): CONFIRMED — `BorderType::Rounded` in `header.rs:115`.
- `>_ GenericAgent` slogan on first interior row: CONFIRMED — `│ >_ GenericAgent`.
- Leading space inside left border (`│ >_ …`): CONFIRMED — `header.rs:121-126` insets one column, padding confirmed.
- model / directory / session each on own row: CONFIRMED — 5 interior rows.
- `/llm switch` on the model row: CONFIRMED.

---

## CLAIM 2 — Model row shows `codex-pro` and `gpt-5.5` (NOT MixinSession)

**Verdict: CONFIRMED**

Rendered evidence from `--dump-frame normal`:
```
│ model:   codex-pro · gpt-5.5   /llm switch
```

The seed at `main.rs:222-223` injects `llm_name="codex-pro"` / `model_real="gpt-5.5"` via `Status` event. `header.rs:36-47` (`model_identity`) reads `app.llm_name` / `app.model_real` preferentially, falling back to the full chain only when absent. The full chain `MixinSession/codex-pro|gpt-5.2|…|kiro` does NOT appear in the header. Unit test `header_box_uses_round4_identity_rows` asserts this explicitly.

---

## CLAIM 3 — Tool calls render as bordered boxes

**Verdict: CONFIRMED**

Rendered evidence from `--dump-frame normal` (tool call row):
```
╭─ web_scan  ✕ error  ·t1 ─────────────────────────────────────────────────────────────────────────╮
│ {"tabs_only": true}                                                                              │
│ 3 tabs scanned · ok                                                                              │
│ !!!Error: SSE overloaded                                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

- Top border matches `╭─ <name>  <badge>  ·tN ─…─╮`: CONFIRMED.
- `web_scan` (name, bold) on top border: CONFIRMED.
- `✕ error` (status badge, error-colored): CONFIRMED.
- `·t1` (dim turn-id tag) on top border: CONFIRMED.
- Arg hint `{"tabs_only": true}` is FIRST interior row: CONFIRMED — `chip.rs:288-290`.
- `[Info]` meta prefix stripped; "3 tabs scanned · ok" in result row: CONFIRMED — `chip.rs:515-527`.
- `!!!Error: SSE overloaded` interior row: CONFIRMED.
- Bottom border `╰─…─╯`: CONFIRMED.
- `… +N more` when truncated: CONFIRMED by unit test `result_overflow_folds_inside_the_box` (10-line result, max_preview=3 → interior carries `… +7 more`). Not visible in dump-frame fixture (only 3 result lines, within max_preview=4).
- Every box row is exactly `inner` (= terminal width) cells wide: CONFIRMED by `cjk_box_rows_stay_exact_width` and by the pre-clip logic in `chip.rs:277,285,305-313`.

---

## CLAIM 4 — Markdown headings bold/colored, NO literal `#`; narration ` ▸ <summary>`

### Sub-claim 4a: Markdown headings bold/colored, no literal `#`

**Verdict: CONFIRMED**

Code proof: `markdown/render.rs:311-319` — `Tag::Heading` handler pushes BOLD + color onto the style stack; it emits **no `#` text at all**. `pulldown_cmark` parses ATX headings (`# …`) and emits only `Start(Heading)` + `Text("Title")` + `End(Heading)` — the `#` prefix is consumed by the parser. No `#` glyph can reach the output. Unit test `md_headings_and_inline_styles` verifies BOLD modifier and content.

### Sub-claim 4b: Narration shows ` ▸ <summary>` (accent triangle + dim)

**Verdict: GAP — two defects**

**What "narration" means:** folded-turn one-line headers (tui_v3 term; each COMPLETED turn collapses to one `▸ title` line). The ACTIVE turn's summary uses `↳ <crumb>` (intentional tui_v4 breadcrumb — not a gap). The reference screenshot shows `▸` lines for folded turns in orange/accent color.

**GAP 1 — `▸` uses DIM color instead of ACCENT:**

`markdown/mod.rs:164-170` renders the entire fold header as ONE `Span` with `Token::Dim`:
```rust
Line::from(Span::styled(
    fold_header_line(title, width),  // "▸ title"
    Style::default()
        .fg(theme.color(crate::theme::Token::Dim))  // WRONG: should be split
        .add_modifier(Modifier::ITALIC),
))
```
The tui_v3 spec (`R1_tui_v3_render.md:181`) and the reference screenshot both show `▸` in `_ACCENT` (orange, `Token::Claude`) with the title in `Token::Dim`. The implementation paints both in gray (`Token::Dim` = `rgb(80,80,80)`).

Fix: split into two spans — `Span::styled("▸ ", accent)` + `Span::styled(title_only, dim_italic)`.
File: `src/markdown/mod.rs:164-172`.

**GAP 2 — No leading space before `▸` (vs spec's ` ▸ title`):**

`fold_header_line()` at `markdown/mod.rs:241-242`:
```rust
let prefix = "▸ ";  // no leading space
```
The tui_v3 spec uses `_indent_rows([head], w)` which prepends ONE space, so the visual is ` ▸ title`. The implementation emits `▸ title` flush-left (no indent). The reference screenshot confirms the leading space: each narration line starts with `▸` after a visible 1-cell indent.

Fix: change `prefix` to `" ▸ "` and adjust `avail = (width as usize).saturating_sub(3)`.
File: `src/markdown/mod.rs:241-243`.

**Dump-frame coverage note (adversarial):** all dump-frame scenarios inject only a 1-turn fixture, so no `▸` fold headers appear in any rendered frame. The `▸` path is exercised only by unit tests (`fold_summary`, `cockpit_folds_completed_turn_to_one_line`) which check the plain text content but NOT the style/color of individual spans. A styled-frame scan with a 2-turn fixture would expose GAP 1 definitively.

---

## CLAIM 5 — Box borders stay aligned at narrow AND wide widths (no wrap-break)

**Verdict: CONFIRMED** (with one documented edge case)

**Header box at narrow widths:**
`header.rs:127` uses `Paragraph::new(rows)` without `.wrap()`. In ratatui, `Paragraph` defaults to NO wrapping — interior lines clip at the available width. The `Block` border is rendered independently of inner content, so the `╭/╰` corners remain aligned regardless of content length. This is ratatui's built-in guarantee.

**Chip box at all widths ≥ 8:**
`chip.rs:277`: `let inner = (width as usize).max(MIN_BOX_INNER)` where `MIN_BOX_INNER = 8`. Every chip row (top, interior, bottom) is pre-sized to exactly `inner` cells via `top_border()`, `interior_row()`, and `border_line()`. The markdown layer then passes these through `wrap_styled_line(line, width)` (markdown/mod.rs:206). Because chip rows are exactly `width` cells, no soft-wrapping occurs. Confirmed by `cjk_box_rows_stay_exact_width` at 24-wide.

**Edge case — terminal narrower than 8 cells:** when `width < MIN_BOX_INNER (8)`, chip rows are 8 cells but `wrap_styled_line` wraps at `width` < 8, producing multiple visual rows per border line and visually breaking the box. This is an extreme edge case (unusable terminal) and not a practical concern, but it is unguarded.

---

## Summary

| Claim | Verdict |
|-------|---------|
| 1 — HEADER rounded multi-line box | CONFIRMED |
| 2 — codex-pro / gpt-5.5 identity | CONFIRMED |
| 3 — Tool call bordered box format | CONFIRMED |
| 4a — Headings: bold, no literal `#` | CONFIRMED |
| 4b — Narration ` ▸ <summary>` | **GAP** (two defects, see below) |
| 5 — Box border alignment narrow/wide | CONFIRMED (edge at <8 cells) |

## GAPS requiring fix

**GAP-4b-1** (render fidelity — color): `src/markdown/mod.rs:164-172` — fold header `▸` uses `Token::Dim` for the entire span; spec requires `Token::Claude` (accent/orange) for `▸` and `Token::Dim` for the title. Fix: emit two spans.

**GAP-4b-2** (layout — indent): `src/markdown/mod.rs:241-242` — `fold_header_line` prefix is `"▸ "` (no leading space); tui_v3 spec and reference screenshot show ` ▸ title` with 1-cell leading indent. Fix: change prefix to `" ▸ "` and adjust `avail` by 1.
