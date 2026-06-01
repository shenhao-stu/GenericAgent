# R1 — tui_v3 render recon → tui_v4 port spec

Reverse-engineered from `frontends/tui_v3.py` (5573 lines). Reference look to
reproduce: `D:/Screenshots/clip_20260601_134922.png` (tui_v3, GOOD). Look to
AVOID: `D:/Screenshots/clip_20260601_134942.png` (tui_v4 current, BAD).

Shared ANSI palette (tui_v3.py:1500-1525, non-Apple branch):
- `_RST = \x1b[0m`
- `_DIM = \x1b[2m` (gray; on Apple_Terminal `\x1b[38;5;244m`)
- `_ACCENT = \x1b[38;2;94;106;210m` (Linear lavender #5e6ad2) → tui_v4 `Token::Claude`/accent
- `_BORDER = \x1b[38;5;146m` (light lavender) → box borders
- `_BOLD = \x1b[1m`
- `_OK = \x1b[38;5;71m` (green), `_ERR = \x1b[38;5;167m` (red)
- box glyphs: rounded corners `╭ ╮ ╰ ╯`, horizontal `─`, vertical `│`
- `_border(left,right,width,style)` = `style + left + '─'*(width-2) + right + _RST` (tui_v3.py:1548-1552)
- `_boxln(plain,colored,w,border)` = `border│ + colored(fit inner=w-4) + pad + border │` (tui_v3.py:2536-2544)

================================================================================
## ITEM 1 — Multi-line HEADER block (slogan + model/dir/session in a ROUNDED box)
================================================================================

### (a) tui_v3 source mechanism
`SB._make_banner_lines(self, w)` — **tui_v3.py:3671-3696**.

Literal construction (tui_v3.py:3682-3692):
```python
rows = [(_ACCENT + '>_' + _RST + ' GenericAgent', '>_ GenericAgent'),
        ('', ''),
        (f'{d}{lbl_model}{r}       {name}   {d}{llm_hint}{r}',
         f'{lbl_model}       {name}   {llm_hint}'),
        (f'{d}{lbl_dir}{r}   {cwd}',  f'{lbl_dir}   {cwd}'),
        (f'{d}{lbl_sess}{r}     {sess_val}', f'{lbl_sess}     {sess_val}')]
top = _border('╭', '╮', w)            # full-width rounded top
bot = _border('╰', '╯', w)            # full-width rounded bottom
lines = ['', top]
lines += [self._boxln(p, c, w) for c, p in rows]   # each row → '│ … │'
lines += [bot, '', f'  {d}{tip(self._tip_idx)}{r}', '']
```
Labels are i18n (tui_v3.py:233-237 / 483-487): `model:` / `directory:` /
`session:`; `banner.session.single` = `single · scrollback` (zh `单会话 · scrollback`);
`banner.llm_hint` = `/llm switch` (zh `/llm 切换`). `cwd` = `os.getcwd()` with
`$HOME`→`~` (tui_v3.py:3675). `name` = `bridge.llm_name` (the full pipe-chain
`MixinSession/codex-pro|getoken_20x|…|kiro`). The banner is its own `Block(kind='banner')`
re-rendered on resize (tui_v3.py:3636-3637, Block doc 1124-1135).

### (b) exact target glyphs / layout (5 rows inside one rounded box, width w)
```
(blank line)
╭────────────────────────────────────────────────────────────────────────╮
│ >_ GenericAgent                                                          │   ← '>_' in _ACCENT, rest plain
│                                                                          │   ← blank interior row
│ model:       <full-pipe-chain>   /llm switch                             │   ← labels _DIM, /llm switch _DIM
│ directory:   D:\GenericAgent                                             │
│ session:     single · scrollback                                         │
╰────────────────────────────────────────────────────────────────────────╯
(blank line)
  <rotating tip in _DIM>
(blank line)
```
- Slogan glyph is literally `>_ GenericAgent` (NOT `❯❯`). Only `>_` is accent-colored.
- model / directory / session each on their OWN interior row (left label dim,
  value default fg). `/llm switch` is appended to the model row, dim, after the
  value (it is NOT right-aligned to the box edge — it trails the model value
  separated by 3 spaces; screenshot confirms it sits just right of the chain).
- Spacing baked into the format strings: `model:` + 7 spaces, `directory:` + 3
  spaces, `session:` + 5 spaces (so the values left-align into a column).
- Box is FULL terminal width (`_border(.., w)`); interior padded by `_boxln` to width.

### (c) tui_v4 file:function to rewrite
**`frontends/tuiapp_v4/src/components/cockpit/header.rs`** → `render_header()`
(currently header.rs:22-54). The BUG: it renders ONE single `Paragraph` line
cramming `❯❯ GenericAgent  llm X  model Y  dir Z  session W` (header.rs:16,37-53).
Rewrite to emit a multi-row ROUNDED-border `Block`/box widget:
- replace `SLOGAN = "❯❯ GenericAgent"` (header.rs:16) with `>_ GenericAgent` (only `>_` accented).
- The header `Rect` must grow to ~7-8 rows (1 blank + 6 box rows + tip). Caller
  is the cockpit layout — check `components/cockpit/mod.rs` for the header height
  constant and bump it; `main.rs` layout split must give header.rs its taller area.
- Draw `ratatui::widgets::Block::default().borders(ALL).border_type(BorderType::Rounded)`
  styled with the border lavender, then render 4 interior `Line`s (slogan, blank,
  model+`/llm switch`, directory, session) as a `Paragraph` inside the block's inner area.
- Keep the FULL model pipe-chain (do NOT call `truncate_model`/`llm_channel` — the
  v3 banner shows the whole `bridge.llm_name` chain). `llm_channel`/`truncate_model`
  belong to the footer, not the header.

================================================================================
## ITEM 2 — TOOL-CALL rendering as a BORDERED BOX
================================================================================

### (a) tui_v3 source mechanism
`_chip_box(tid_str, combo, st, w, result='')` — **tui_v3.py:1728-1775**.
Fed by `_render_assistant` (tui_v3.py:4885-4905) which finds `▸ tN name hint · status`
placeholders (regex `_CHIP_PLACEHOLDER_RE`, tui_v3.py:1685) emitted by `_compress`
(tui_v3.py:4849-4882, line 4865: `\n▸ t{tid} {name} {hint} · {st}\n`) and calls
`_chip_box` DIRECTLY (bypassing markdown) so the box edges never wrap-break.

Status mapping (tui_v3.py:1740-1741):
```python
sti, stcol = (('✓ ok', _OK) if st=='ok' else
              ('✕ error', _ERR) if st=='error' else ('· …', _DIM))
tag = f'·t{tid_str}'
```
Wide-box branch (inner ≥ 24), tui_v3.py:1753-1774 — THE literal:
```python
header_c = (' ' + _BOLD+name+_RST + '  ' + stcol+sti+_RST + '  ' + _DIM+tag+_RST + ' ')
fill = max(1, inner - 3 - cell_len(header_plain))
top = _ACCENT + '╭─' + _RST + header_c + _ACCENT + '─'*fill + '╮' + _RST   # name ON the top border
bot = _border('╰', '╯', inner, _ACCENT)
content_w = max(1, inner - 4)
body_rows = _wrap_cells(hint, content_w)            # args/hint first
body_rows += _result_preview(result, 4, content_w)  # then result, ≤4 rows
for ch in body_rows:                                # each interior row:
    out.append(_ACCENT+'│'+_RST + ' ' + _DIM+ch+_RST + ' '*pad + ' ' + _ACCENT+'│'+_RST)
out.append(bot)
```
Narrow branch (inner < 24): no box, just `name ✓ ok ·tN` then dim body lines (tui_v3.py:1744-1752).

`_arg_hint` (tui_v3.py:1654-1682): pulls ONE line from args JSON (priority keys
`command/script/path/file_path/url/query/question`), file paths shortened to
`…/basename`, clipped to 60 cells. `_result_preview` (tui_v3.py:1690-1725): first
N content lines, skips `[Action]/[Status]/[Stdout]` meta, unwraps JSON envelopes
(`stdout/output/result/content/text`), clips each to row width with `…`, appends
`… +K more` when truncated.

### (b) exact target glyphs / layout
```
╭─ web_scan  ✓ ok  ·t6 ─────────────────────────────────────────────────────╮
│ {"switch_tab_id": "", "tabs_only": false}                                  │   ← hint (args) row, _DIM
│ 1337214814                                                                 │   ← result preview row, _DIM
╰────────────────────────────────────────────────────────────────────────────╯
```
- Top border: `╭─` (accent) + ` name ` (BOLD) + `  ` + status badge + `  ` + `·tN`
  (dim) + ` ` + `─…─` (accent) + `╮`. Tool name + status + turn-id ALL live ON the
  top border line.
- Status badge glyphs: `✓ ok` (green) / `✕ error` (red) / `· …` (dim pending).
- turn-id token = literal `·t6` (dot + `t` + number), dim.
- Interior rows: `│ ` + content (dim) + pad + ` │`, borders accent.
- args (hint) shown FIRST, then result preview (≤4 rows wide-box / ≤3 narrow).
- Long result folds via `… +N more` (tui_v3.py:1722-1724).
- Bottom border: `╰────╯` accent, full inner width.
- Screenshot match: the box in clip_134922 shows `web_scan · … ·t6` on the top
  border and `1337214814` inside — confirms pending status badge `· …` + body.

### (c) tui_v4 file:function to rewrite
**`frontends/tuiapp_v4/src/render/chip.rs`** → `render_chip_bullet()` (chip.rs:272-316)
and its struct `ChipBullet` (chip.rs:319-352). THE BUG: it produces a FLAT bullet —
`⏺ name  {raw json}` + 2-col-indented result, explicitly "NO box" (chip.rs:1-22,
257-261; test `bullet_glyph_and_indented_result` even ASSERTS no `╭│╰`, chip.rs:473-477).
This is exactly the inferior look in clip_134942 (`○ web_scan {"switch_tab_id":…}`).

Rewrite plan:
1. In `render/chip.rs`: replace `render_chip_bullet` with a `render_chip_box(call, width, max_preview, expanded) -> ChipBox`
   that returns the materialized box ROWS (top border w/ name+badge+`·tN` on it,
   interior arg+result rows wrapped to `inner-4`, bottom border). Reuse the EXISTING
   pure helpers already in this file: `tool_status` (chip.rs:95), `parse_tool_calls`
   (chip.rs:142), `clip_cells` (chip.rs:361). Keep status badge text from
   `ToolStatus::badge()` (chip.rs:80-86) but change glyphs to `✓ ok` / `✕ error` / `· …`
   and add a `·tN` turn-id field to `ToolCall` (or pass tid in). Status COLOR token
   from `ToolStatus::token()` (chip.rs:69-76: Success/Error/Dim).
2. The companion consumer **`frontends/tuiapp_v4/src/markdown/mod.rs`** →
   `push_tool_bullet()` (mod.rs:373-420, called at mod.rs:348) currently lays the
   bullet out flat (head span at mod.rs:390-400, indented result rows mod.rs:406-419).
   Rewrite to `push_tool_box`: emit the top-border `Line` (accent corners + bold name
   + colored badge + dim `·tN` + accent dashes), each interior row as `│ … │` (accent
   border spans + dim body span), bottom-border `Line`. Keep the `NodeId::Tool`
   fold-affordance tagging (mod.rs:346-347,387) so click-to-expand still works; the
   `▸ +N more` / `▾` affordance moves INSIDE the box as its own interior row.
3. Border color = accent (`Token::Claude`), matching v3's `_chip_box` use of `_ACCENT`
   for the corners/verticals (not the dimmer banner `_BORDER`).
4. Delete/adjust the chip.rs tests that assert "no box glyphs" (chip.rs:473-477) and
   the bullet-glyph asserts (chip.rs:466-491, 503-547) to expect the box form.

================================================================================
## ITEM 3 — Intermediate THINKING / narration steps (gray italic, small arrow)
================================================================================

### (a) tui_v3 source mechanism
Per-turn `<summary>` titles, collapsed into folded one-line headers.
`_render_block` assistant fold path — **tui_v3.py:3608-3635**, the literal at
**tui_v3.py:3625**:
```python
head = _ACCENT + '▸ ' + _RST + _DIM + seg['title'] + _RST
out.extend(_indent_rows([head], w))     # _indent_rows = prepend one leading space
```
Title extraction in `_fold_turns` (tui_v3.py:3582-3595): strips fenced code +
`<thinking>` (`_TITLE_CLEAN_RE`, tui_v3.py:1112), takes the first `<summary>…</summary>`
(`_SUMMARY_PERTURN_RE`, tui_v3.py:1111) else the first content line, trims `args:` tail,
clips to 72 chars + `...`. Shown only when `self._fold_all` is True (default True,
tui_v3.py:2253) and the assistant block has ≥1 non-final turn (so each completed
turn collapses to ONE narration line; the final/current turn stays full text).
Live incremental fold trigger: tui_v3.py:5028-5032.
(The plan-card also uses `  ▸ ` for the current-step line, tui_v3.py:2498/2529 —
distinct: that is indented 2 + dim and lives in the plan box, not the transcript.)

### (b) exact target glyphs / layout
```
 ▸ 准备新开Trending页查看
 ▸ 无Trending页，打开新标签
 ▸ 已开新页，扫描Trending
```
- prefix glyph = `▸ ` (BLACK RIGHT-POINTING SMALL TRIANGLE U+25B8) + space, in `_ACCENT`.
- title text in `_DIM` (gray). Screenshot's "italic" impression = the dim gray weight.
- one leading space indent (from `_indent_rows`).
- one line PER completed (non-final) turn; emitted in scrollback above the
  current/last turn's prose + its tool boxes.

### (c) tui_v4 file:function to rewrite
**`frontends/tuiapp_v4/src/render/fold.rs`** (per-turn fold/summary collapse;
already calls `chip::find_turn_line`/`parse_tool_calls` at fold.rs:314,333 and is the
fold engine). Ensure the collapsed per-turn header renders as ` ▸ <summary>` with
`▸` in accent (`Token::Claude`/Suggestion) + title in `Token::Dim`. Verify the
glyph is `▸` (U+25B8) and the title is the `<summary>` first-line (clip 72). The
summary strip is already wired in `markdown/mod.rs` (`body_no_summary`, mod.rs:326);
the FOLDED-header styling for non-final turns is the fold.rs concern. (No new file —
this is a styling/glyph confirmation in the existing fold path.)

================================================================================
## ITEM 4 — PET + gerund line near the composer (bear face + gerund + ellipsis)
================================================================================

### (a) tui_v3 source mechanism
`_live_lines` after-block builder — **tui_v3.py:3269-3274**, the literal:
```python
if self._running and self._asking is None:
    el = time.time() - self._t0_anchor
    after.append(' ' + _heat(el) + _pet(el, self._spin // 5) + _RST +
                 '  ' + _DIM + _gerund(el) + '…' + _RST)
```
- `_pet(el, frame)` (tui_v3.py:1910-1918): picks a heat-tier (0:<20s 1:<60s 2:<180s
  3:≥180s) then a 4-frame blink cycle. Default style `bear` (tui_v3.py:1907). Bear
  frames `_PETS_BEAR` (tui_v3.py:1892-1897): tier0 `('ʕ•ᴥ•ʔ','ʕ-ᴥ-ʔ','ʕ•ᴥ•ʔ','ʕ•ᴥ-ʔ')`,
  escalating to `ʕ>ᴥ<ʔ`/`ʕ@ᴥ@ʔ`/`ʕTᴥTʔ` at the stressed tier.
- `_heat(el)` (tui_v3.py:1840-1846): mint `rgb(170,232,170)` <20s → amber
  `rgb(212,167,44)` <60s → orange `rgb(220,107,31)` <180s → bold red `rgb(255,44,44)`.
  The PET (and the leading `_SPIN` braille on the status line) carry the heat color.
- `_gerund(el)` (tui_v3.py:1854-1855): `_GERUNDS[int(el//6) % len]`, rotates every 6s;
  pool `SPINNER_GERUNDS` (tui_v3.py:62-71): Pondering / Untangling / Unraveling / …
- frame cadence: `self._spin // 5` → pet swaps ~every 0.5s (`_spin` ticks 0.1s, tui_v3.py:4745).

### (b) exact target glyphs / layout
```
 ʕ•ᴥ•ʔ  Untangling…
```
- leading space; bear face in heat color; TWO spaces; gerund in `_DIM`; trailing `…`.
- screenshot clip_134922 shows exactly `ʕ-•o-ʔ  Untangling…` (bear, focused tier).

### (c) tui_v4 file:function to rewrite
**`frontends/tuiapp_v4/src/components/cockpit/footer.rs`** → `render_spinner()`
(footer.rs:39-90). It ALREADY composes a pet + glyph + gerund LEAD (footer.rs:52-62
via `flavor::pet` / `flavor::gerund`, heat via `heat_token`/`heat_bold` footer.rs:43-47).
DIVERGENCE to fix: v4 forces a STATIC `⠿` glyph between pet and gerund (footer.rs:42,61)
and a CC-style ` (secs · ↑in ↓out · ctx · effort)` trailer (footer.rs:69-87). The v3
"pet line near composer" is JUST `<space><pet>  <gerund>…` with NO braille glyph and
NO token trailer (that data lives on the separate status/footer line, item 5). Either
split this into a dedicated above-composer pet line matching tui_v3.py:3273-3274, or
drop the `⠿` + `(…)` trailer from this lead. Pet styles/heat already ported in
`src/flavor/` (`PetStyle`, `pet`, `gerund`, `heat_token`, `heat_bold`) — verify the
BEAR frames + 6s gerund rotation + 0.5s blink cadence match tui_v3.py:1892-1897/1854.

================================================================================
## ITEM 5 — FOOTER status line (braille + [branch] + model-chain + gerund + elapsed + tok/s)
================================================================================

### (a) tui_v3 source mechanism
`SB._status_line(self, w)` — **tui_v3.py:2360-2375**, the literal return:
```python
return f'[main] {name} │ {state}{cost}'
```
- `name` = `bridge.llm_name` (full pipe-chain).
- `state` (tui_v3.py:2362-2373): if asking → `status.asking`; elif running →
  `f'{_gerund(el)} {_elapsed(el)}{tps}'` + ` · Esc stop`; else `○ ready`.
- `tps` (tui_v3.py:2366-2370): ` · {rate:.0f} tok/s` where rate ≈ `len(stream)/4/el`.
- `_elapsed` (tui_v3.py:2131-2134): `{int(s)}s` under 60s, else `m:ss`.
- `cost` from `_cost_str` (tui_v3.py:2189-2205): ` │ ctx ▰▱▱… pct% (used/cap) · N tok`.

The BRAILLE LEAD is prepended where the status line is placed in `_live_lines`
(tui_v3.py:3309-3313), the literal at 3310-3311:
```python
lead = _heat(el) + _SPIN[self._spin % len(_SPIN)] + ' ' + _RST   # braille spinner, heat-colored
after.append(lead + _DIM + _clip_cells(self._status_line(w), max(2, w-2)) + _RST)
```
`_SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'` (tui_v3.py:1921) — cycles at 0.1s; the WHOLE status string after
the lead is `_DIM` (gray). Idle path (not running): plain `_DIM` status, no braille
(tui_v3.py:3313).

### (b) exact target glyphs / layout
Running (screenshot clip_134922 bottom row):
```
⠿ [main] MixinSession/codex-pro|getoken_20x|getoken|anyrouter_chenyt|tabcode_claude|tabcode_kiro|kiro │ Untangling 46s · 52 tok/s
```
- braille spinner glyph (one of `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`, heat-colored) + space
- `[main]` literal (the leading branch-ish tag; v3 hardcodes `[main]`, NOT git branch)
- full model pipe-chain
- ` │ ` separator (dim)
- gerund + elapsed (`Untangling 46s`) + ` · {n} tok/s`
- whole line dim after the braille lead.

### (c) tui_v4 file:function to rewrite
**`frontends/tuiapp_v4/src/components/cockpit/footer.rs`** → `render_session_info()`
(footer.rs:123-157). THE BUG (clip_134942 bottom): it renders
`llm · model · effort · ctx · branch` with `llm_channel`+`truncate_model` (footer.rs:126-144)
— a DIFFERENT shape from v3's `[main] <full-chain> │ <gerund> <elapsed> · tok/s`.
Rewrite to match `_status_line`:
- lead with the heat-colored braille spinner (cycle `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` by `now_ms/100`,
  NOT the static `⠿`) when busy; omit when idle.
- `[main]` literal tag (or wire the real branch but keep the `[...]` bracket form).
- the FULL model pipe-chain (`app.model`), NOT `truncate_model`/`llm_channel`.
- ` │ ` then gerund + elapsed + ` · {tok/s}` while running; ` │ ○ ready` when idle.
- the whole post-lead string in `Token::Dim`.
- `tok/s` = live rate (app has tok counts + elapsed; compute `len/4/elapsed` analog
  or a tok/s field). The ctx-bar `▰▱` belongs to the `cost` tail (footer.rs `ctx_bar`
  at footer.rs:181-185 already exists — append it as ` │ ctx ▰▱ pct%`).
NOTE: `render_session_info` is the "[main] … │ state" footer; the pet/gerund LEAD
(item 4) is the SEPARATE above-composer line — do not merge the two.

================================================================================
## TARGET FILE SUMMARY (5 tui_v4 files)
================================================================================
1. `src/components/cockpit/header.rs` — `render_header` → multi-row rounded box
   (`>_ GenericAgent` slogan + model/directory/session each on own line + `/llm switch`).
2. `src/render/chip.rs` — `render_chip_bullet`/`ChipBullet` → rounded `_chip_box`
   (name+`✓ ok`+`·tN` on the top border; args+result inside `│ … │`; `… +N more` fold).
3. `src/markdown/mod.rs` — `push_tool_bullet` (mod.rs:373-420) → `push_tool_box`
   (emits the box border `Line`s + interior `│`-bracketed rows; keeps NodeId fold tag).
4. `src/components/cockpit/footer.rs` — `render_spinner` (item 4 pet line: drop the
   `⠿` glyph + `(…)` trailer, keep `<space><pet>  <gerund>…`) AND `render_session_info`
   (item 5 status line: `⠿/braille [main] <full-chain> │ <gerund> <elapsed> · tok/s`).
5. `src/render/fold.rs` — per-turn `<summary>` collapse → ` ▸ <title>` (accent `▸` U+25B8 + dim title).
