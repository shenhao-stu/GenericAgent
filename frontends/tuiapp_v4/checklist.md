# tui_v4 — Round-6 checklist (LIVE tracker)

> Verification rule (R4/R5/R6): a box ticks ONLY on **live/styled** evidence (text fed
> through `apply_bridge_event`, styled `TestBackend` cells scanned, real key/mouse
> events, or `--dump-frame` rendered rows) — never a clean-fixture unit test alone.
> Safety (R5 freeze): every cargo gate `timeout`-wrapped; ONE cargo at a time; impl on
> Opus; every new `while`/`loop` proven to advance/break. Specs: `recon/round6/R1_audit.md`
> (parity), `recon/round6/R2_reference.md` (codex/cc rendering), `IMPLEMENTATION_PLAN_R6.md`.
> **Result: v0.6.0 — 383 passed / 0 failed / 0 warnings / 1.4s; 4 parity invariants green;
> release exe built; adversarial Monitor (5 Opus reviewers) — M1/M2/M4/M5 CONFIRMED,
> M3 found one GA-core blocker (S5, see below).**

---

## Slices

- [x] **S1 — content + composer HUG THE TOP.** Round-5 bottom-anchor flipped: `split_cockpit`
  pins the transcript to `Length(total_visual_lines)` + a trailing `Min(0)` bottom spacer
  (overflow → `Min(0)` + composer pinned, scrolls). `prepare_frame` syncs the width-only
  cache BEFORE the split so both `split_cockpit` calls read the same `total_visual_lines()`
  (no geometry drift; 4 parity invariants intact). `render_transcript` bottom-anchor removed.
  *Dump-verified (normal/busy/done): content at TOP, blank at BOTTOM.* **Monitor M1 CONFIRMED.**
- [x] **S2 — remove the ugly duplicate `↳` breadcrumb (R1 GAP A).** `render_turn_body` keeps
  `hoist_summary` (strips raw `<summary>` tags) but no longer pushes the `↳ <summary>` line —
  the ` ▾ `/` ▸ ` fold header is the single canonical summary display (re-collapse intact).
  *Dump: `▾` header once, zero `↳`.* **Monitor M1 CONFIRMED.**
- [x] **S3 — `/goal /hive /conductor` command-char FX (like `/morphling`).** All four style
  the command word with strong, per-char, phase-0-visible, mutually-distinct RGB+BOLD
  effects (Goal gradient / Hive swarm / Conductor baton / Morphling rainbow); the effect now
  shows WHILE TYPING (inverse caret rides one grapheme inside the styled word).
  **Monitor M2 CONFIRMED** (helper-level test; wired into the live render path).
- [x] **S4 — footer pipe fields + ctx bar.** `Channel: | Model: | Effort: | ctx: [█…░] Nk/Mk (P%) | branch: | mouse:`.
  Additive `context_used`/`context_limit` wired ga_bridge→protocol→app→reducer (wire-safe).
  **Gating fixed inline:** the ctx BAR is the headline → dropped LAST (trailing `mouse:`/`branch:`
  shed first). *Dump (100w): `ctx: [████████░░░░░░░░] 96k/200k (48%)`.* **Monitor M2 CONFIRMED.**
- [x] **S5 — paste IMAGES + FILES into the composer (the tuiapp_v2/tui_v3 PATH model).**
  M3 found the base64→`Submit.images` route dies at GA core (`agentmain.py run()` drops
  `task["images"]`). The user steered to the v2/v3 approach: **the image/file travels as its
  PATH inline in the prompt; GA reads the file** (tuiapp_v2 `submit_user_message` sends only
  text — `put_task(text)` — the path is in it). Reworked: Ctrl+V folds a clipboard bitmap →
  temp PNG `[Image #N]` (arboard image-data + png 0.17) / a file path → `[File #N]`; on submit
  BOTH expand to their PATH inline (`paste.rs expand`). Dropped the base64/`Submit.images`
  route + its dead helpers (`collect_images`/`build_submit_images`/`mime_for_path`). **No
  GA-core edit — fully in-constraint, v2/v3 parity.** *Test: `attach_image_folds_placeholder_and_submit_inlines_path`.*
- [x] **S6 — port `continue_cmd.py` preview fix → Rust.** Dirty-summary rejection
  (`=== `/`"role"`/>200ch) + newest-first `last_user`/`prompt_blocks` fallback (no more JSON
  debris). **Monitor M4 CONFIRMED** (malformed-log test).
- [x] **S7 — `/verbose` interactive inspector (R1 GAP B, parity red line).** Flat `Vec<String>`
  → `Vec<ToolAuditRecord>{name,args,result,status,raw}` from real wire tool calls; two-pane
  list+detail; ↑↓/kj select, PgUp/PgDn scroll, Enter cycle result→args→raw, c copy, e export.
  **Monitor M4 CONFIRMED.**

## Audit (task 1a) — R1: commands + keybindings are a COMPLETE SUPERSET of v2∪v3 (0 gaps);
the only parity gaps were the 2 rendering items above (A=S2 done, B=S7 done).

## Upstream sync (task 5) — `lsdefine/main` merged (`continue_cmd.py` fix → S6).

## Safety (Monitor M5 CONFIRMED) — the R5 freeze cannot recur: 0 new bare loops, the two
legacy `while`s advance/break at EOF, `markdown/render.rs` untouched, all 10 `--dump-frame`
scenarios exit 0, `context_win*3` ctx truth intact, no GA-core file edited by these slices.

## Pending
- [ ] Push + tag `tui-v4-v0.6.0` → release CI (authorized this round).

> Round-6 lesson (write to memory): when a TUI feature seems to need a GA-core change,
> check how tuiapp_v2/tui_v3 already do it FIRST — image/file paste is **path-in-prompt**
> (`put_task(text)` only; GA reads the file), NOT base64 multimodal. The base64/`Submit.images`
> route would have forced an `agentmain.py` red-line edit for nothing.
