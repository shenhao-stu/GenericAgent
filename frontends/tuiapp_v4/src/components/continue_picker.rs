//! components/continue_picker.rs — the `/continue` SEARCHABLE session picker
//! (checklist §4 `/continue`: "searchable picker over session logs — content-grep
//! + lazy load, v2").
//!
//! It lists past sessions from `temp/model_responses/model_responses_*.txt`
//! (newest first, with a cheap head/tail preview + a completed-round count) and a
//! SEARCH box: typing filters the list by an AND-term, case-insensitive content
//! grep (the basename + preview match first; only otherwise do we LAZY-LOAD a
//! bounded head window of the file and grep its bytes — mirroring
//! `continue_cmd.search_sessions` / `file_contains_all` so 17 MB logs never stall
//! the UI). Enter RESTORES the selected log into the active session's history via
//! the EXISTING restore path (`Command{name:"restore", args:<path>}` →
//! `ga_bridge.handle_restore` → `continue_cmd.restore`).
//!
//! LOAD-BEARING (the `continue_search_filter` deliverable): the session listing +
//! the content-grep filter + the selection/window logic are PURE functions over
//! injectable data (a session slice + a "read this file's head" closure), so they
//! are unit-tested with no disk and no live python. The renderer only PAINTS.

use std::path::{Path, PathBuf};

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::components::picker::{window_slice, PICKER_ROWS};
use crate::i18n::{self, Lang};
use crate::theme::{Theme, Token};

/// Content-grep budget: read at most this many bytes from the HEAD of a log when
/// filtering (the lazy-load window). Mirrors `continue_cmd._GREP_WIN` (1 MiB) — the
/// user-typed prompt + first reply + early summaries live here, which is what users
/// recall sessions by, and it bounds memory regardless of file size.
pub const GREP_WIN: usize = 1024 * 1024;

/// Preview window (head/tail) for the cheap listing preview. Mirrors
/// `continue_cmd._PREVIEW_WIN` (32 KiB).
pub const PREVIEW_WIN: usize = 32 * 1024;

/// One past session row: the log path + its metadata for the picker. PURE data;
/// the heavy file scan that builds it is in [`list_sessions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinueSession {
    /// Absolute path to the `model_responses_*.txt` log (the restore target).
    pub path: PathBuf,
    /// File mtime (seconds) — for newest-first ordering.
    pub mtime: u64,
    /// A cheap preview line (last `<summary>` else first user-ish line).
    pub preview: String,
    /// Completed Prompt→Response round count (a cheap header-pair tally).
    pub rounds: u32,
}

impl ContinueSession {
    /// The log file's basename (shown as the row's secondary id).
    pub fn basename(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

/// The `/continue` overlay: the (newest-first) session list, a search buffer, the
/// FILTERED view + selection. Typing edits `query`; the filtered indices recompute
/// against `sessions` via the lazy content-grep; Enter restores the selected log.
#[derive(Debug, Clone)]
pub struct ContinuePicker {
    /// All discovered sessions, newest-first (the unfiltered universe).
    pub sessions: Vec<ContinueSession>,
    /// The live search buffer (AND terms, case-insensitive).
    pub query: String,
    /// Indices into `sessions` that currently match `query` (recomputed on edit).
    pub filtered: Vec<usize>,
    /// The highlighted row, indexing `filtered`.
    pub sel: usize,
}

impl ContinuePicker {
    /// Build the picker from a session list (already newest-first). The filter
    /// starts empty (all sessions shown), selection at the top.
    pub fn new(sessions: Vec<ContinueSession>) -> Self {
        let filtered = (0..sessions.len()).collect();
        ContinuePicker { sessions, query: String::new(), filtered, sel: 0 }
    }

    /// True if there are NO sessions at all to continue (the empty-state).
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Number of currently-matching rows.
    pub fn matches(&self) -> usize {
        self.filtered.len()
    }

    /// The session under the current selection (if any).
    pub fn selected(&self) -> Option<&ContinueSession> {
        self.filtered.get(self.sel).and_then(|&i| self.sessions.get(i))
    }

    /// The selected session's restore PATH (the `Command{restore}` arg). `None` if
    /// nothing matches the current search.
    pub fn selected_path(&self) -> Option<String> {
        self.selected().map(|s| s.path.to_string_lossy().into_owned())
    }

    /// Move the selection by `delta` over the FILTERED rows (clamped). PURE.
    pub fn move_sel(&mut self, delta: isize) {
        self.sel = crate::commands::registry::move_sel(self.sel, delta, self.filtered.len());
    }

    /// Type a char into the search buffer + re-filter (lazy content-grep). The
    /// selection re-clamps to the new match count. `read_head` lazily yields a
    /// file's head window (only invoked for sessions whose meta doesn't already
    /// match) so a real call passes a disk reader and a test passes a fake.
    pub fn type_char(&mut self, c: char, read_head: impl Fn(&Path) -> Option<Vec<u8>>) {
        self.query.push(c);
        self.refilter(read_head);
    }

    /// Backspace the search buffer + re-filter.
    pub fn backspace(&mut self, read_head: impl Fn(&Path) -> Option<Vec<u8>>) {
        self.query.pop();
        self.refilter(read_head);
    }

    /// Recompute `filtered` against the current `query` and re-clamp the selection.
    /// PURE-ish (the only effect is the injected `read_head`, used lazily).
    pub fn refilter(&mut self, read_head: impl Fn(&Path) -> Option<Vec<u8>>) {
        self.filtered = filter_sessions(&self.query, &self.sessions, &read_head);
        if self.sel >= self.filtered.len() {
            self.sel = self.filtered.len().saturating_sub(1);
        }
    }

    /// The visible window `(start, slice_of_filtered_indices)` for the scroll
    /// viewport (PICKER_ROWS rows around the selection). PURE.
    pub fn window(&self) -> (usize, &[usize]) {
        window_slice(&self.filtered, self.sel, PICKER_ROWS)
    }
}

/// Filter `sessions` by an AND-term, case-insensitive content grep, returning the
/// matching INDICES (order preserved = newest-first). The rule mirrors
/// `continue_cmd.search_sessions`:
///   * an empty / whitespace query keeps everything;
///   * a session matches if its META (basename + preview) contains every term,
///   * OR (LAZY) the head window of its file contains every term.
/// `read_head` is invoked only for the lazy branch (so meta-matches cost no I/O).
/// PURE over the injected reader — the `continue_search_filter` deliverable pins it.
pub fn filter_sessions(
    query: &str,
    sessions: &[ContinueSession],
    read_head: &impl Fn(&Path) -> Option<Vec<u8>>,
) -> Vec<usize> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return (0..sessions.len()).collect();
    }
    let terms: Vec<String> = q.split_whitespace().map(|t| t.to_string()).collect();
    if terms.is_empty() {
        return (0..sessions.len()).collect();
    }
    let mut out = Vec::new();
    for (i, s) in sessions.iter().enumerate() {
        // 1. Cheap META match (basename + preview), no I/O.
        let meta = format!("{}\n{}", s.basename(), s.preview).to_lowercase();
        if terms.iter().all(|t| meta.contains(t)) {
            out.push(i);
            continue;
        }
        // 2. LAZY content grep: read the head window only now.
        if let Some(bytes) = read_head(&s.path) {
            if bytes_contain_all(&bytes, &terms) {
                out.push(i);
            }
        }
    }
    out
}

/// True iff the (lowercased) `bytes` contain EVERY term (case-insensitive). Mirrors
/// `continue_cmd.file_contains_all` — bytes-level so it stays in a fixed memory
/// envelope and avoids UTF-8 cost. An empty buffer matches nothing. PURE.
pub fn bytes_contain_all(bytes: &[u8], terms: &[String]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let hay = String::from_utf8_lossy(bytes).to_lowercase();
    terms.iter().all(|t| t.is_empty() || hay.contains(t))
}

/// The `temp/model_responses` log directory under a GA repo root (where
/// `model_responses_*.txt` live — `continue_cmd._LOG_DIR`).
pub fn log_dir(repo_root: &Path) -> PathBuf {
    repo_root.join("temp").join("model_responses")
}

/// Discover past sessions under `repo_root`: glob `temp/model_responses/
/// model_responses_*.txt`, skip tiny/empty files, build a cheap head/tail preview +
/// a completed-round tally, and sort newest-first. Effectful (reads the dir +
/// per-file head/tail windows — NOT the whole file, so it's bounded). `exclude` is
/// a substring (e.g. this process's own log basename) to omit. Degrades to empty on
/// any error.
pub fn list_sessions(repo_root: &Path, exclude: Option<&str>) -> Vec<ContinueSession> {
    let dir = log_dir(repo_root);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<ContinueSession> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().map(|n| n.to_string_lossy().into_owned()) {
            Some(n) => n,
            None => continue,
        };
        // Only `model_responses_*.txt` (and not snapshots' transient suffixes are
        // fine to include — they are restorable too).
        if !(name.starts_with("model_responses_") && name.ends_with(".txt")) {
            continue;
        }
        if let Some(ex) = exclude {
            if name.contains(ex) {
                continue;
            }
        }
        let Ok(meta) = entry.metadata() else { continue };
        let size = meta.len();
        if size < 32 {
            continue; // skip empty/placeholder logs (continue_cmd's sz<32 rule).
        }
        let mtime = meta
            .modified()
            .ok()
            .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let (preview, rounds) = preview_and_rounds(&path, size);
        if preview.is_empty() {
            continue;
        }
        out.push(ContinueSession { path, mtime, preview, rounds });
    }
    // Newest-first.
    out.sort_by(|a, b| b.mtime.cmp(&a.mtime));
    out
}

/// Read a bounded head/tail of a log to extract a preview line + a completed-round
/// count (cheap: only the head + tail windows, never the whole file). Effectful.
fn preview_and_rounds(path: &Path, size: u64) -> (String, u32) {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else {
        return (String::new(), 0);
    };
    // Head window for the preview + the round tally.
    let head_len = (size as usize).min(GREP_WIN);
    let mut head = vec![0u8; head_len];
    let read = f.read(&mut head).unwrap_or(0);
    head.truncate(read);
    // Tail window for a trailing <summary> (the most informative preview).
    let mut tail = Vec::new();
    if size as usize > PREVIEW_WIN {
        if f.seek(SeekFrom::End(-(PREVIEW_WIN as i64))).is_ok() {
            let _ = f.take(PREVIEW_WIN as u64).read_to_end(&mut tail);
        }
    }
    let preview = preview_from_windows(&head, &tail);
    let rounds = count_rounds(&head);
    (preview, rounds)
}

/// Build the preview from head + tail byte windows: the LAST `<summary>…</summary>`
/// in the tail, else the first "real" line (not a `===`/`###` header or a
/// `<history>` block) in the head. Mirrors `continue_cmd._preview_from_file`. PURE.
pub fn preview_from_windows(head: &[u8], tail: &[u8]) -> String {
    let tail_s = String::from_utf8_lossy(tail);
    if let Some(sum) = last_summary(&tail_s) {
        return collapse_ws(&sum);
    }
    // Also check the head's tail for a summary in case the file was small (head==all).
    let head_s = String::from_utf8_lossy(head);
    if let Some(sum) = last_summary(&head_s) {
        return collapse_ws(&sum);
    }
    for line in head_s.lines() {
        let s = line.trim();
        if !s.is_empty()
            && !s.starts_with("===")
            && !s.starts_with("###")
            && !s.contains("<history>")
        {
            return collapse_ws(s);
        }
    }
    String::new()
}

/// The last `<summary>…</summary>` body in `s` (the freshest task summary). PURE.
fn last_summary(s: &str) -> Option<String> {
    let mut last: Option<String> = None;
    let mut rest = s;
    while let Some(open) = rest.find("<summary>") {
        let after = &rest[open + "<summary>".len()..];
        if let Some(close) = after.find("</summary>") {
            let body = after[..close].trim();
            if !body.is_empty() {
                last = Some(body.to_string());
            }
            rest = &after[close + "</summary>".len()..];
        } else {
            break;
        }
    }
    last
}

/// Count completed Prompt→Response pairs by header lines (a Prompt header followed
/// by a Response header = one round). Mirrors `continue_cmd
/// ._count_complete_rounds_from_file` over the head window. PURE.
pub fn count_rounds(head: &[u8]) -> u32 {
    let s = String::from_utf8_lossy(head);
    let mut pending = false;
    let mut rounds = 0u32;
    for line in s.lines() {
        if line.starts_with("=== Prompt ===") {
            pending = true;
        } else if line.starts_with("=== Response ===") && pending {
            rounds += 1;
            pending = false;
        }
    }
    rounds
}

/// Collapse internal whitespace runs to single spaces (a one-line preview). PURE.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A disk reader for the lazy content-grep: the head window (≤ [`GREP_WIN`]) of a
/// file, or `None` on error. The live picker passes this to `refilter`; tests pass
/// a fake map. Effectful.
pub fn read_head_window(path: &Path) -> Option<Vec<u8>> {
    use std::io::Read;
    let f = std::fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    f.take(GREP_WIN as u64).read_to_end(&mut buf).ok()?;
    Some(buf)
}

// ---------------------------------------------------------------------------
// Renderer (PAINTS only).
// ---------------------------------------------------------------------------

/// Draw the `/continue` searchable picker: a search box at the top, then the
/// filtered, scrolling session list. Centered, bordered, theme-tokened.
pub fn render(frame: &mut Frame, area: Rect, picker: &ContinuePicker, theme: &Theme, lang: Lang) {
    let w = (area.width.saturating_sub(6)).clamp(40, 96);
    let visible = picker.matches().min(PICKER_ROWS);
    let h = (visible as u16 + 6).min(area.height.saturating_sub(2)).max(8);
    let card = centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.color(Token::Claude)))
        .title(Span::styled(
            format!(" {} ", i18n::t(lang, "continue.title")),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(card);
    frame.render_widget(block, card);
    let inner_w = inner.width as usize;

    let mut lines: Vec<Line> = Vec::new();
    // The search box (always shows the live query + a cursor cell).
    let mut search: Vec<Span> = vec![Span::styled(
        i18n::t(lang, "continue.search"),
        Style::default().fg(theme.color(Token::Suggestion)),
    )];
    if picker.query.is_empty() {
        search.push(Span::styled(
            i18n::t(lang, "continue.search.placeholder"),
            Style::default().fg(theme.color(Token::Dim)),
        ));
    } else {
        search.push(Span::styled(
            picker.query.clone(),
            Style::default().fg(theme.color(Token::Text)),
        ));
    }
    search.push(Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)));
    lines.push(Line::from(search));
    lines.push(Line::from(""));

    if picker.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  {}", i18n::t(lang, "continue.empty")),
            Style::default().fg(theme.color(Token::Dim)),
        )));
    } else if picker.matches() == 0 {
        lines.push(Line::from(Span::styled(
            format!("  {}", i18n::t(lang, "continue.no_match")),
            Style::default().fg(theme.color(Token::Dim)),
        )));
    } else {
        let (start, slice) = picker.window();
        for (row, &idx) in slice.iter().enumerate() {
            let s = &picker.sessions[idx];
            let selected = start + row == picker.sel;
            let mut spans: Vec<Span> = vec![Span::styled(
                if selected { "❯ " } else { "  " },
                Style::default().fg(theme.color(Token::Suggestion)),
            )];
            // round count + preview (preview clipped to fit).
            spans.push(Span::styled(
                format!("[{}{}] ", s.rounds, rounds_suffix(lang)),
                Style::default().fg(theme.color(Token::Dim)),
            ));
            let preview_w = inner_w.saturating_sub(14);
            spans.push(Span::styled(
                super::clip_to(&s.preview, preview_w),
                Style::default()
                    .fg(theme.color(if selected { Token::Suggestion } else { Token::Text }))
                    .add_modifier(if selected { Modifier::BOLD } else { Modifier::empty() }),
            ));
            lines.push(Line::from(spans));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        i18n::t(lang, "continue.hint"),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The localized "rounds" suffix shown after a session's round count.
fn rounds_suffix(lang: Lang) -> &'static str {
    i18n::t(lang, "continue.rounds")
}

/// A centered card `Rect` (clamped to `area`). PURE geometry.
fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sess(name: &str, mtime: u64, preview: &str, rounds: u32) -> ContinueSession {
        ContinueSession {
            path: PathBuf::from(format!("/temp/model_responses/{name}")),
            mtime,
            preview: preview.into(),
            rounds,
        }
    }

    /// THE deliverable test: the `/continue` content-grep search filter.
    /// An empty query keeps everything; a term matches via META (basename/preview)
    /// with NO file read; a term that's only in the file BODY matches via the LAZY
    /// head-window read; multiple terms are ANDed; selection re-clamps as the match
    /// set shrinks. The reader is injected so the test needs no disk.
    #[test]
    fn continue_search_filter() {
        let sessions = vec![
            sess("model_responses_111.txt", 300, "refactor the parser", 4),
            sess("model_responses_222.txt", 200, "fix the login bug", 7),
            sess("model_responses_333.txt", 100, "write release notes", 2),
        ];
        // A fake "file body" store: only session 333's BODY mentions "kubernetes"
        // (its preview does NOT), so it can only be found by the lazy content grep.
        let read_head = |p: &Path| -> Option<Vec<u8>> {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let body = match name.as_str() {
                "model_responses_111.txt" => "=== Prompt ===\nrefactor the parser\nLALR tables",
                "model_responses_222.txt" => "=== Prompt ===\nfix the login bug\nsession cookie",
                "model_responses_333.txt" => "=== Prompt ===\nwrite release notes\ndeploy to kubernetes cluster",
                _ => "",
            };
            Some(body.as_bytes().to_vec())
        };

        // 1. Empty query → all three, in order.
        assert_eq!(filter_sessions("", &sessions, &read_head), vec![0, 1, 2]);
        assert_eq!(filter_sessions("   ", &sessions, &read_head), vec![0, 1, 2]);

        // 2. META match (preview), case-insensitive: "LOGIN" → session 222 only.
        assert_eq!(filter_sessions("LOGIN", &sessions, &read_head), vec![1]);
        // basename match: the pid digits "333".
        assert_eq!(filter_sessions("333", &sessions, &read_head), vec![2]);

        // 3. BODY-only match via lazy content grep: "kubernetes" is in 333's file
        // body but NOT its preview/basename → found only by reading the head.
        assert_eq!(filter_sessions("kubernetes", &sessions, &read_head), vec![2]);

        // 4. AND terms: "release notes" → 333 (both in preview); "deploy kubernetes"
        // → 333 (both in body).
        assert_eq!(filter_sessions("release notes", &sessions, &read_head), vec![2]);
        assert_eq!(filter_sessions("deploy kubernetes", &sessions, &read_head), vec![2]);
        // A term present in NO session → empty.
        assert!(filter_sessions("nonexistent-term-xyz", &sessions, &read_head).is_empty());

        // 5. The picker integrates it: typing narrows + re-clamps the selection.
        let mut p = ContinuePicker::new(sessions.clone());
        assert_eq!(p.matches(), 3);
        p.sel = 2; // select the last row.
        // Type "login" → only session 222 matches; selection re-clamps to 0.
        for c in "login".chars() {
            p.type_char(c, &read_head);
        }
        assert_eq!(p.matches(), 1);
        assert_eq!(p.sel, 0);
        assert_eq!(p.selected().unwrap().preview, "fix the login bug");
        assert_eq!(p.selected_path().as_deref(), Some("/temp/model_responses/model_responses_222.txt"));
        // Backspacing widens the match set again.
        p.backspace(&read_head); // "logi"
        // "logi" still only matches 222's preview ("login").
        assert_eq!(p.matches(), 1);
        while !p.query.is_empty() {
            p.backspace(&read_head);
        }
        assert_eq!(p.matches(), 3, "clearing the query restores all sessions");
    }

    /// The preview extractor prefers a trailing `<summary>`, else the first real
    /// head line; the round counter tallies Prompt→Response header pairs. PURE.
    #[test]
    fn preview_and_round_parsing_is_pure() {
        // A trailing summary wins (from the tail window).
        let head = b"=== Prompt ===\nhello there\n=== Response ===\n[...]";
        let tail = b"some text <summary>Did the thing successfully</summary> trailing";
        assert_eq!(preview_from_windows(head, tail), "Did the thing successfully");

        // No summary → the first non-header, non-history line of the head.
        let head2 = b"=== Prompt ===\n### [WORKING MEMORY]\nsummarize the repo\n=== Response ===";
        // `###` lines are skipped; `=== Prompt ===` skipped → "summarize the repo".
        assert_eq!(preview_from_windows(head2, b""), "summarize the repo");

        // Round count: 2 complete Prompt→Response pairs; a trailing lone Prompt
        // (in-flight) is NOT counted.
        let log = b"=== Prompt ===\na\n=== Response ===\nb\n=== Prompt ===\nc\n=== Response ===\nd\n=== Prompt ===\ne";
        assert_eq!(count_rounds(log), 2);
        assert_eq!(count_rounds(b""), 0);

        // bytes_contain_all: AND, case-insensitive; empty buffer matches nothing.
        let terms = vec!["alpha".to_string(), "beta".to_string()];
        assert!(bytes_contain_all(b"has ALPHA and BeTa here", &terms));
        assert!(!bytes_contain_all(b"only alpha", &terms));
        assert!(!bytes_contain_all(b"", &terms));
    }

    /// The renderer paints the search box + a row to an in-memory backend.
    #[test]
    fn continue_renders_search_and_rows() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let theme = Theme::ga_default();
        let p = ContinuePicker::new(vec![
            sess("model_responses_111.txt", 300, "refactor the parser", 4),
        ]);
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render(f, f.area(), &p, &theme, Lang::En)).unwrap();
        let buf = term.backend().buffer();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("Continue"), "paints the title");
        assert!(text.contains("refactor the parser"), "paints a session row");
    }
}
