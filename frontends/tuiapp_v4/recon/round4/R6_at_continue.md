# R6 — Fix spec: `@` file-mention completeness + `/continue` search

Recon investigator pass. **No source was edited.** All cites are `file:line` at HEAD.
Scope: `frontends/tuiapp_v4/src` (Rust+ratatui) with `frontends/tuiapp_v2.py`,
`frontends/continue_cmd.py`, and `frontends/tuiapp_v4/scripts/ga_bridge.py` as
references.

---

## FEATURE 1 — `@` file-mention is not loading/showing ALL content

### 1.1 The `@` pipeline as built (trace)

1. **Query detection** — `src/input/mod.rs:164` `Composer::at_query()` → delegates to
   `src/input/file_expand.rs:39` `at_query(text, caret)` (pure span finder).
2. **Candidate universe** — `src/app/mod.rs:649` `AppState::list_project_files()` returns
   an `Arc<Vec<String>>`, memoized in the `file_index` cache
   (`src/app/mod.rs:223`, a `RefCell<Option<(Instant, Arc<Vec<String>>)>>`), TTL =
   `FILE_INDEX_TTL` 5s (`src/app/mod.rs:44`). Cache miss re-walks via
   `src/input/paths.rs:211` `list_project_files(root)`.
3. **Ranking** — `src/input/file_expand.rs:95` `rank_files(partial, files)` scores +
   sorts, then **truncates to `MAX_PICKER_ROWS`** (`file_expand.rs:128`,
   const at `file_expand.rs:133` = **8**).
4. **Render** — `src/components/cockpit/dropdown.rs:45-49` `render_dropdown` →
   `render_file_picker` (`dropdown.rs:93`), which `.take(MAX_PICKER_ROWS)` again
   (`dropdown.rs:103`). Height: `dropdown_height` (`dropdown.rs:18-35`).
5. **Keyboard** — `src/input/keymap.rs:336-358` (Up/Down/Tab/Enter over `ranked`),
   completion via `Composer::complete_file` (`src/input/mod.rs:563`) →
   `apply_pick` (`file_expand.rs:76`).
6. **Submit-time inline** — `AppState::expand_at_paths` (`src/app/mod.rs:636`) →
   `expand_file_refs` (`file_expand.rs:167`); per-file cap `MAX_INLINE_BYTES` 100 KB
   (`file_expand.rs:18`, enforced at `file_expand.rs:245`).

### 1.2 Root cause — WHY results are incomplete

There are **three independent caps**, layered. In order of impact:

**(A) The dominant cause — the picker only ever surfaces 8 rows, with NO "+N more"
signal and NO fuzzy widening.** `rank_files` hard-truncates to `MAX_PICKER_ROWS`
(`file_expand.rs:100` for the empty-query branch, `file_expand.rs:128` for the
scored branch; const = 8 at `file_expand.rs:133`). The renderer truncates *again* to
8 (`dropdown.rs:103`). So even when 200 files match `@co`, the user sees 8 and has
no way to know more exist or to scroll to them. There is **no scroll/window** for the
`@` picker (contrast the `/continue` picker which uses `window_slice` /
`PICKER_ROWS`, `continue_picker.rs:141-143`). This is the "not showing ALL" symptom
the user reports while typing.

**(B) Secondary — the walk is capped at 5000 files AND the walk order is
non-deterministic, so on a large repo the indexed *set itself* is a partial,
arbitrary slice.** `MAX_INDEXED_FILES` = 5000 (`paths.rs:38`); the walk breaks at the
cap (`paths.rs:219-221` and `paths.rs:240-242`). The walk is a **stack** (`queue.pop()`,
`paths.rs:218`) fed by `read_dir` in OS order (`paths.rs:225`) → effectively a DFS in
arbitrary directory order. The `out.sort()` (`paths.rs:246`) happens *after* the cap,
so it sorts whichever 5000 happened to be visited — a file that sorts early
alphabetically can be entirely absent if it lived in a late-visited subtree. On this
repo the 5000 cap is realistic: `memory/`, `plugins/`, `frontends/` and nested
clones produce many thousands of files.

**(C) Over-aggressive default skip + no `**` glob.** `DEFAULT_SKIP_DIRS`
(`paths.rs:17-34`) bakes in `temp` (the `model_responses` tree — intentional, Q12) plus
the usual `target/.git/node_modules/...`. That is mostly correct, but it is applied to
*every* segment by basename semantics (`paths.rs:140`), so a legitimately-tracked dir
that happens to be named e.g. `venv`, `.vscode`, `target`, or `temp` anywhere in the
tree is silently dropped even if git tracks it. Also the matcher has **no `**`
support** (`paths.rs:179-181` documents this), so a `.gitignore` with `**/foo` under-
or over-matches vs real git. These are edge contributors, not the main bug.

**Not a cause:** the 5s TTL (`mod.rs:44`) only debounces re-walks; it serves a
*complete-as-of-last-walk* snapshot, not a partial one — the partiality is the 5000
cap + the 8-row truncation, not staleness. The 100 KB `MAX_INLINE_BYTES`
(`file_expand.rs:18`) is a *submit-time* per-file cap and does not affect the picker
listing; it can truncate a single large inlined file's *content* but is out of scope
for "showing all candidates".

### 1.3 How v2 / v3 do `@` completion (reference)

`frontends/tuiapp_v2.py` and `frontends/tui_v3.py` have **no `@`-file picker**. Grep
of `tuiapp_v2.py` for `@`/file-completion shows only the `/continue` and image-paste
flows; the `@path` inline-expand-on-submit concept is the tui_v3 `_expand` behavior
that `file_expand.rs` already ports (see the module docstring `file_expand.rs:1-11`
crediting "tui_v3 §C-46"). **Conclusion: there is no upstream picker UX to copy for
Feature 1** — the design below is the canonical one, modeled on Codex's
`FileSearchManager` (the synchronous analogue is already noted at `mod.rs:220`).

### 1.4 Proposed fix — balance completeness vs performance

Three changes, smallest-blast-radius first. Keep the pure/effectful split intact.

**Change 1 (PRIMARY) — scroll the `@` picker + show a "+N more" tail + keep a
generous in-memory match set.** This fixes symptom (A) without any walk cost.

- `src/input/file_expand.rs:95` `rank_files`: stop truncating to `MAX_PICKER_ROWS`.
  Return the *full ranked* list (it is already scored + sorted). Truncation is a
  *view* concern, not a ranking concern. Optionally cap at a sane upper bound
  (e.g. a new `MAX_RANKED = 500`) so a no-op `@` over a 5000-file index doesn't
  build a huge `Vec` every keystroke. Remove the `.truncate(MAX_PICKER_ROWS)` at
  `file_expand.rs:100` and `.take(MAX_PICKER_ROWS)` at `file_expand.rs:128`.
- `src/components/cockpit/dropdown.rs:93` `render_file_picker`: replace the flat
  `.take(MAX_PICKER_ROWS)` (`dropdown.rs:103`) with a **windowed** view around the
  selection, reusing `crate::components::picker::window_slice(files, sel, MAX_PICKER_ROWS)`
  (same helper the `/continue` picker uses, `continue_picker.rs:28,141`). Add a final
  dim row `"… +N more (keep typing to narrow)"` when `files.len() > MAX_PICKER_ROWS`,
  styled like the existing hint row (`dropdown.rs:118-121`).
- `src/input/keymap.rs:341-348`: Up/Down already wrap modulo `ranked.len()` — with a
  full list this now scrolls through *all* matches (the window in the renderer
  follows `file_sel`). No new field needed; `file_sel` (`src/input/mod.rs:97`) already
  drives it. `dropdown_height` (`dropdown.rs:18-35`) keeps using `MAX_PICKER_ROWS`
  for the row count so the panel height is unchanged (+1 for the "+N more" row).
- **Trade-off:** zero extra I/O; one extra `min`/window per frame; the `Vec` is at
  most `MAX_RANKED`. This alone resolves the user-visible "incomplete" complaint
  because every matching file is now reachable.

**Change 2 (SECONDARY) — make the walk *complete-up-to-cap deterministic* and raise
the cap, with cheap pruning so it stays fast.**

- `src/input/paths.rs:211` `list_project_files`: change the frontier from a LIFO
  `Vec` + `pop()` (`paths.rs:218`) to a **FIFO `VecDeque` (true BFS)** and sort each
  directory's entries before enqueueing, so the first N files are the
  shallowest/alphabetically-first — i.e. the cap drops *deep* files, never
  *near/early* ones. This makes the truncation predictable and far less likely to
  hide a file the user is actually `@`-mentioning (they almost always mean a
  near-root, tracked source file).
- `src/input/paths.rs:38` `MAX_INDEXED_FILES`: raise 5000 → e.g. **20000**. With the
  `temp/target/node_modules` skips (`paths.rs:17-34`) the *tracked* file count of this
  repo is well under that, so in practice nothing is dropped while the bound still
  protects a pathological monorepo. Keep the cap (don't remove it) — an unbounded
  walk on a huge tree is the original freeze (Q12, `mod.rs:215-219`).
- **Trade-off:** BFS over the same skip set costs ~the same as the current DFS; the
  higher cap costs more only on repos that actually have >5000 tracked files, and
  even then the walk is memoized for `FILE_INDEX_TTL`.

**Change 3 (PERF — keep the picker fluid as the index grows) — background/lazy
refresh + a longer TTL, instead of a blocking re-walk on the hot path.**

The current cache re-walks *synchronously inside `list_project_files()`* when the TTL
lapses (`mod.rs:649-658`), so the first keystroke after 5s pays the full walk on the
render thread. With a 20000-file cap that stall is more noticeable. Two options,
pick one:

- **(3a, minimal)** Bump `FILE_INDEX_TTL` (`mod.rs:44`) to e.g. **30s**. The walk is
  cheap relative to a human session and a 30s window still makes a brand-new file
  `@`-completable "within a few seconds" as the doc promises. Lowest risk; no
  threading.
- **(3b, fuller, Codex-parity)** Serve the *stale* snapshot immediately and refresh in
  the background: change `file_index` (`mod.rs:223`) so a TTL-lapsed read returns the
  old `Arc` **now** and spawns one `std::thread` that re-walks and swaps the cache via
  a `Mutex`/channel (async is banned, `mod.rs:219-220`, but a detached worker thread is
  not async). The picker then never blocks; it just shows a ≤TTL-old list. This is the
  true "walk once, hand out a shared snapshot" of Codex's `FileSearchManager`. Higher
  blast radius (the `RefCell` becomes a `Mutex`/`Arc<Mutex<..>>` and the 3 call sites at
  `dropdown.rs:28,46` + `keymap.rs:337` keep their `Arc` deref unchanged).

**Recommended bundle:** Change 1 (mandatory — fixes the actual symptom) + Change 2 +
Change 3a. Change 3b only if profiling shows the re-walk stutter after raising the cap.

**Optional polish (gitignore fidelity):** add `**` handling to the matcher
(`paths.rs:179-206` `glob_match` / `paths.rs:130-143` `rule_matches`) so nested
`**/x` ignore rules match git exactly, and consider reading nested `.gitignore`
files, not just the root one (`paths.rs:213`). These reduce false drops from
over-broad basename matching (cause C) but are lower priority than Changes 1–2.

### 1.5 Tests to add / update

- `file_expand.rs` (`mod tests`, ~`file_expand.rs:310`): `rank_files` no longer
  truncates — assert it returns *all* matches (drop the implicit `<=8` expectation in
  `rank_files_prefers_basename_prefix`, `file_expand.rs:311-330`).
- `paths.rs` (`mod tests`, ~`paths.rs:317`): add a "deep file is dropped before a
  shallow file at the cap" test proving BFS ordering; keep
  `default_skip_excludes_temp` (`paths.rs:278`) green.
- `dropdown.rs`: a render test asserting the "+N more" row appears when matches >
  `MAX_PICKER_ROWS` and the window follows `file_sel`.

---

## FEATURE 2 — `/continue` SEARCH, replicated from tuiapp_v2

### 2.1 KEY FINDING: tui_v4 ALREADY implements the searchable `/continue` picker.

The searchable picker is built end-to-end and matches v2's content-grep semantics.
This feature is **substantially DONE**; the work item is **parity polish**, not a
build-from-scratch. Evidence (full wiring):

| Stage | tui_v4 location |
|---|---|
| Command registered | `src/commands/registry.rs:86` (`continue`, "searchable picker over session logs", `Ui`) |
| Handler builds picker | `src/commands/dispatch.rs:203-214` → `list_sessions` + `Overlay::Continue(ContinuePicker::new(...))` |
| Overlay variant | `src/app/types.rs:48` `Continue(ContinuePicker)` |
| Picker model + filter | `src/components/continue_picker.rs:69-194` (`ContinuePicker`, `filter_sessions`, `bytes_contain_all`) |
| Session discovery | `continue_picker.rs:208-250` `list_sessions` (glob `temp/model_responses/model_responses_*.txt`, preview + round tally, newest-first) |
| Lazy grep reader | `continue_picker.rs:347-353` `read_head_window` (≤ `GREP_WIN` 1 MiB) |
| Input routing | `src/input/views.rs:400-435` — `Char`→`type_char`, `Backspace`→`backspace`, Up/Down→`move_sel`, Enter→`selected_path`, Esc→close |
| Renderer (search box + rows) | `continue_picker.rs:361-438` |
| Restore | `views.rs:416-422` emits `Command{name:"restore", args:<path>}` |

### 2.2 EXACTLY how tuiapp_v2 lets the user search past sessions

- **Where the query is typed:** a Textual `Input` mounted *above* the inner choice
  list inside a `SearchableChoiceList` wrapper (`tuiapp_v2.py:1426-1460`,
  placeholder "Search sessions: type to filter, Esc to cancel", id `continue-search`).
  Only `/continue` opts in: `msg.searchable = True` + `msg.all_choices = list(choices)`
  (`tuiapp_v2.py:3939-3940`); every other picker bypasses this wrapper
  (`tuiapp_v2.py:1429-1431`).
- **What it filters on (BOTH name AND content):** `_filter_choices`
  (`tuiapp_v2.py:1368-1423`). Each whitespace token must hit either the **label**
  (the one-line preview row text) **or the basename of the path** (cheap meta match,
  `tuiapp_v2.py:1406-1410`), **OR** the **session file content** — first ~1 MB via
  `continue_cmd.file_contains_all` (`tuiapp_v2.py:1412-1422`,
  `continue_cmd.py:115-134`). Multi-term = AND, case-insensitive. Empty query →
  full list (`tuiapp_v2.py:1382-1384`).
- **How results update incrementally:** `on_input_changed` (`tuiapp_v2.py:1517-1545`)
  with a **0.22s debounce** (`DEBOUNCE_SEC`, `tuiapp_v2.py:1515`): an empty query
  applies instantly (`tuiapp_v2.py:1532-1535`); a non-empty query schedules one
  deferred `_apply_filter` and **cancels the previous timer** so only the latest
  prefix is grepped (`tuiapp_v2.py:1522-1541`). `_apply_filter`
  (`tuiapp_v2.py:1547-1592`) skips stale rebuilds (`tuiapp_v2.py:1559-1560`),
  recomputes `_filter_choices`, **rebuilds the inner picker** (eager ≤50 rows / lazy
  `LazyChoiceList` >50, `tuiapp_v2.py:1492-1507,1581`), shows a disabled
  "(no matches)" row on empty results (`tuiapp_v2.py:1571-1579`), and re-pins the
  list into view (`tuiapp_v2.py:1592`).
- **Keys while the search box has focus:** `on_key` (`tuiapp_v2.py:1594-1628`)
  redirects Up/Down/PgUp/PgDn/Home/End to the inner picker (replaying one step so the
  first arrow isn't swallowed, `tuiapp_v2.py:1603-1620`) and Enter/Right commits the
  highlighted row (`tuiapp_v2.py:1622-1627`); Esc cancels.
- **What is shown per result:** built in `_cmd_continue`
  (`tuiapp_v2.py:3920-3930`): `"{relative_age} · {session_name} · {N}轮 · {preview≤50}"`,
  value = the log path. Preview = last `<summary>` else first user line
  (`continue_cmd.py:164-182`); round count via cached header-pair tally
  (`continue_cmd.py:233-271`).
- **Restore on select:** `on_select` → `_do_continue_restore(path)`
  (`tuiapp_v2.py:3933,3951`) → `reset_conversation` + `continue_cmd.restore` +
  `continue_extract` replays bubbles (`tuiapp_v2.py:3953-3981`). v2 *also* supports
  `/continue N` and `/continue <name>` non-interactive forms
  (`tuiapp_v2.py:3894-3916`) — these are not part of the *search box*.

### 2.3 tui_v4 side — what exists vs the v2 reference (gap analysis)

The Rust port already mirrors the v2/continue_cmd semantics deliberately
(`continue_picker.rs:1-18` docstring; the deliverable test
`continue_search_filter` at `continue_picker.rs:473-533` proves AND-term,
meta-vs-lazy-content, and selection re-clamp):

- META (basename+preview) match with no I/O, then LAZY content grep — exactly
  `search_sessions` (`continue_picker.rs:154-183` vs `continue_cmd.py:137-161`).
- `bytes_contain_all` byte-level AND grep — `continue_cmd.file_contains_all`
  (`continue_picker.rs:188-194` vs `continue_cmd.py:115-134`).
- 1 MiB `GREP_WIN` + 32 KiB `PREVIEW_WIN` — same constants
  (`continue_picker.rs:36,40` vs `continue_cmd.py:106,112`).
- Restore goes through the identical existing path: `Command{restore}`
  (`views.rs:421`) → `ga_bridge.handle_restore` (`scripts/ga_bridge.py:738-789`) →
  `continue_cmd.restore` + `extract_ui_messages` replay (`ga_bridge.py:763-789`),
  arg-shape robust via `_restore_path_from_args` (`ga_bridge.py:791-798`), dispatched
  at `ga_bridge.py:931-934`.

**Remaining gaps vs v2 (the actual deliverable):**

1. **No debounce on the content grep.** `views.rs:430` calls
   `picker.type_char(c, read_head_window)` → `refilter` (`continue_picker.rs:119-137`)
   on *every keystroke*, grepping up to 1 MiB × N sessions synchronously on the render
   thread. v2 explicitly added a 0.22s debounce because per-keystroke grep felt laggy
   (`tuiapp_v2.py:1509-1515`). On ~270 sessions this is the one behavioral regression.
2. **Per-result row lacks the relative age / session name** that v2 shows. v4 shows
   `[N轮] preview` only (`continue_picker.rs:418-429`); v2 shows
   `age · name · N轮 · preview` (`tuiapp_v2.py:3929`).
3. **No `/continue N` / `/continue <name>` non-interactive forms.** v4's handler
   ignores `cmd.args` and always opens the picker (`dispatch.rs:203-213`); v2 supports
   both (`tuiapp_v2.py:3894-3916`). (Lower priority — the *search box* is the ask.)
4. **Residual dead path:** `PickerKind::Continue` (`src/components/picker.rs:64`,
   `dispatch.rs:408-417`) is a legacy generic-picker branch NOT used by `/continue`
   anymore (live path is `Overlay::Continue`). It only survives as a title-key + a unit
   test (`picker.rs:549`). Leave it or delete it; it does not affect Feature 2.

### 2.4 Proposed design (close the gaps; high-cohesion / low-coupling)

Keep the pure model in `continue_picker.rs` (it already isolates filter logic from the
renderer and from disk via the injected `read_head` closure — that is the
high-cohesion/low-coupling seam to preserve). Changes:

**Gap 1 — debounce the grep (the important one).** Two options:

- **(1a, recommended, no threads)** Make typing cheap by default and only run the
  *lazy content* branch after a short quiet period — mirror v2's "empty applies
  instantly, non-empty debounces":
  - Add a `last_edit: Instant` (or a tick counter) to `ContinuePicker`
    (`continue_picker.rs:69-79`). `type_char`/`backspace` update `query` and stamp
    `last_edit`, then run **only the META filter** immediately (cheap, no I/O) — split
    `filter_sessions` (`continue_picker.rs:154`) into `filter_meta_only` and the
    existing full grep, or pass a flag. The full content grep runs from the event
    loop's tick once `last_edit` is ≥ ~0.2s old (a `maybe_refilter(now)` the main loop
    calls each tick, analogous to how effects advance on `tick()`, `mod.rs:36`). This
    requires the `/continue` overlay to be ticked — wire a `maybe_refilter` call where
    the main loop already ticks overlays.
  - Trade-off: meta matches (basename/preview) are instant; content matches land ~0.2s
    after the user pauses — exactly v2's feel — with zero new threads and the pure
    filter untouched in spirit.
- **(1b, simplest)** Bound the cost instead of debouncing: lower `GREP_WIN`
  (`continue_picker.rs:36`) for the *interactive* path or cap the number of sessions
  that get the lazy grep per refilter. Less faithful to v2 but a one-line mitigation if
  the tick plumbing is unwanted.

**Gap 2 — richer rows (cheap, pure).** Add `mtime`-derived relative age to the row and
(optionally) a session name. `ContinueSession` already carries `mtime`
(`continue_picker.rs:49`). Add a pure `rel_age(mtime, now) -> String` (port
`continue_cmd._rel_time`, `continue_cmd.py:17-22`) and prepend it in the row spans
(`continue_picker.rs:418-429`), so the row reads `age · [N轮] preview` — matching v2's
information density. Session-name parity needs a `session_names` lookup (no Rust
equivalent today); treat as optional/out-of-scope unless required.

**Gap 3 (optional) — `/continue N`.** In `dispatch.rs:203`, if `cmd.args` is a bare
integer, index `list_sessions(...)` and emit `Command{restore, args:<path>}` directly
(skip the overlay), mirroring `tuiapp_v2.py:3897-3903`. Name-based `/continue <name>`
needs `session_names` and is out of scope.

**No `ga_bridge.py` changes are required** for the search box: restore already works
end-to-end (`ga_bridge.py:738-798`, dispatched `ga_bridge.py:931-934`). `ga_bridge`
would only be touched if you wanted server-side session *enumeration* (it has none
today; v4 lists locally in Rust via `continue_picker::list_sessions`, which is the
correct low-coupling choice — the UI owns its own listing, the bridge owns restore).

### 2.5 Tests

- Extend `continue_search_filter` (`continue_picker.rs:473`) for the debounce split:
  meta-only filter is synchronous; content match only after `maybe_refilter`.
- `rel_age` pure unit test (boundaries: <60s, <3600s, <86400s, days).
- Renderer test asserting the age prefix appears (extend
  `continue_renders_search_and_rows`, `continue_picker.rs:563`).

---

## Summary of file:line touch points

**Feature 1 (must):** `file_expand.rs:95-133` (stop truncating in `rank_files`,
`MAX_PICKER_ROWS` stays a *view* const), `dropdown.rs:93-123` (window + "+N more"),
`keymap.rs:341-348` (scroll over full list — already correct). **Should:**
`paths.rs:211-248` (BFS + sort), `paths.rs:38` (cap 5000→20000), `mod.rs:44` (TTL
5s→30s) or `mod.rs:223,649-658` (background refresh).

**Feature 2 (already wired; parity polish):** `continue_picker.rs:69-79,119-137,154`
(debounce/meta-split), `continue_picker.rs:418-429` (+`rel_age`),
`dispatch.rs:203-214` (optional `/continue N`). No `ga_bridge.py` change needed.
