//! input/file_expand.rs — the `@path` magic prefix (checklist §4; tui_v3 §C-46).
//! Two behaviors:
//!   * **on submit**: each in-root `@relative/path` token expands to the inlined
//!     file contents as `[File: p]\n<content>\n[/File]` (files < 100 KB).
//!   * **while typing**: when the caret sits in an `@query`, a fuzzy file PICKER
//!     ranks project files (gitignore-aware) so Tab/Enter completes the path.
//!
//! The query-span detection, ranking, and pick-apply are PURE over (text, caret,
//! file list) so the whole flow is unit-testable without a TTY or disk. The
//! actual filesystem read (`expand_file_refs`) and the gitignore-aware walk live
//! behind small effectful wrappers; the directory walk's FILTER is pure + tested.

use std::path::Path;

use unicode_segmentation::UnicodeSegmentation;

/// Max bytes a single `@path` inline will read (checklist: "< 100 KB").
pub const MAX_INLINE_BYTES: u64 = 100 * 1024;

/// An active `@` query under the caret: the byte span `[start, end)` of the
/// `@...` token and the partial path typed so far (without the leading `@`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtQuery {
    /// Byte offset of the `@`.
    pub start: usize,
    /// Byte offset of the caret (end of the partial token).
    pub end: usize,
    /// The partial path after the `@` (may be empty right after typing `@`).
    pub partial: String,
}

/// Detect an `@query` under the caret. Returns `Some` iff the caret sits inside
/// (or just after) an `@...` token whose body contains no whitespace. PURE.
///
/// The token starts at the most recent `@` at/ before the caret that is at the
/// start of input or preceded by whitespace (so an email `a@b` is NOT a query),
/// and runs to the caret. If any whitespace appears between that `@` and the
/// caret, there is no active query.
pub fn at_query(text: &str, caret: usize) -> Option<AtQuery> {
    let caret = caret.min(text.len());
    let head = &text[..caret];
    // Find the last '@' in head.
    let at = head.rfind('@')?;
    // The char before '@' must be whitespace or start-of-text (avoid `user@host`).
    if at > 0 {
        let prev = head[..at].chars().next_back();
        if let Some(c) = prev {
            if !c.is_whitespace() {
                return None;
            }
        }
    }
    let partial = &head[at + 1..];
    // A whitespace in the partial means the token already ended.
    if partial.chars().any(|c| c.is_whitespace()) {
        return None;
    }
    Some(AtQuery {
        start: at,
        end: caret,
        partial: partial.to_string(),
    })
}

/// The result of applying a picked path over an [`AtQuery`] span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickResult {
    pub text: String,
    /// New caret byte offset (just after the inserted `@path` + a trailing space).
    pub caret: usize,
}

/// Splice the chosen relative `path` over the active `@` query span, leaving
/// `@path ` (with a trailing space so the token is closed and the picker hides)
/// and the caret after it. PURE.
pub fn apply_pick(text: &str, q: &AtQuery, path: &str) -> PickResult {
    let before = &text[..q.start];
    let after = &text[q.end..];
    let token = format!("@{path} ");
    let caret = before.len() + token.len();
    let mut out = String::with_capacity(before.len() + token.len() + after.len());
    out.push_str(before);
    out.push_str(&token);
    out.push_str(after);
    PickResult { text: out, caret }
}

/// Rank candidate `files` (relative paths) against the `partial` query with a
/// simple, deterministic fuzzy/substring scorer (lower = better), returning the
/// best matches first. PURE so the picker order is unit-testable.
///
/// Scoring (tui_v3-style, conservative): an empty query keeps natural order; a
/// basename prefix match beats a basename substring beats a full-path substring;
/// ties break by shorter path then lexically. Non-matches are dropped.
pub fn rank_files(partial: &str, files: &[String]) -> Vec<String> {
    let q = partial.to_ascii_lowercase();
    if q.is_empty() {
        let mut out = files.to_vec();
        out.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        out.truncate(MAX_PICKER_ROWS);
        return out;
    }
    let mut scored: Vec<(u32, &String)> = Vec::new();
    for f in files {
        let lf = f.to_ascii_lowercase();
        let base = lf.rsplit(['/', '\\']).next().unwrap_or(&lf);
        let score = if base.starts_with(&q) {
            0
        } else if base.contains(&q) {
            1
        } else if lf.contains(&q) {
            2
        } else if subsequence(&q, &lf) {
            3
        } else {
            continue;
        };
        scored.push((score, f));
    }
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.len().cmp(&b.1.len()))
            .then_with(|| a.1.cmp(b.1))
    });
    scored
        .into_iter()
        .map(|(_, f)| f.clone())
        .take(MAX_PICKER_ROWS)
        .collect()
}

/// Max rows the `@` picker shows at once.
pub const MAX_PICKER_ROWS: usize = 8;

/// True if `needle` is an (in-order) subsequence of `hay` — the loosest fuzzy
/// match, used as the last-resort tier. Both lowercased by the caller.
fn subsequence(needle: &str, hay: &str) -> bool {
    let mut it = hay.chars();
    for nc in needle.chars() {
        loop {
            match it.next() {
                Some(hc) if hc == nc => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}

/// The outcome of expanding the `@path` tokens in a submitted message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandResult {
    /// The message text with each in-root `@path` replaced by an inlined block.
    pub text: String,
    /// The relative paths that were actually inlined (for a status / audit).
    pub inlined: Vec<String>,
}

/// Expand in-root `@relative/path` tokens in `text` into inlined file blocks,
/// reading each file (< 100 KB) under `root`. A token whose file is missing,
/// too large, or escapes the root is left verbatim (never silently dropped).
///
/// This is the effectful submit-time expander; the token SCAN + the block
/// FORMAT are pure helpers ([`find_at_tokens`], [`format_inline`]) so the shape
/// is tested without disk.
pub fn expand_file_refs(text: &str, root: &Path) -> ExpandResult {
    let tokens = find_at_tokens(text);
    if tokens.is_empty() {
        return ExpandResult {
            text: text.to_string(),
            inlined: Vec::new(),
        };
    }
    let mut out = String::with_capacity(text.len());
    let mut inlined = Vec::new();
    let mut cursor = 0usize;
    for (start, end, rel) in tokens {
        out.push_str(&text[cursor..start]);
        match read_in_root(root, &rel) {
            Some(content) => {
                out.push_str(&format_inline(&rel, &content));
                inlined.push(rel);
            }
            None => out.push_str(&text[start..end]), // leave token verbatim.
        }
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    ExpandResult { text: out, inlined }
}

/// Find every `@path` token in `text` as `(start, end, relative_path)`. A token
/// starts at an `@` that is at start-of-text or preceded by whitespace and runs
/// until the next whitespace. PURE.
pub fn find_at_tokens(text: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'@' {
            let at_start = i == 0 || text[..i].chars().next_back().is_some_and(|c| c.is_whitespace());
            if at_start {
                // Read until whitespace.
                let mut j = i + 1;
                while j < bytes.len() && !text[j..].chars().next().unwrap().is_whitespace() {
                    j += text[j..].chars().next().unwrap().len_utf8();
                }
                let rel = &text[i + 1..j];
                if !rel.is_empty() {
                    out.push((i, j, rel.to_string()));
                }
                i = j;
                continue;
            }
        }
        i += text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
    }
    out
}

/// Format one inlined file block: `[File: p]\n<content>\n[/File]` (tui_v3
/// `_expand`). PURE.
pub fn format_inline(rel: &str, content: &str) -> String {
    format!("[File: {rel}]\n{content}\n[/File]")
}

/// Read a file at `root/rel` if it stays inside `root` and is < 100 KB. Returns
/// `None` (the token is left verbatim) on any failure. The path-escape guard is
/// lexical (rejects `..` segments) — conservative but safe for a chat inline.
fn read_in_root(root: &Path, rel: &str) -> Option<String> {
    // Reject absolute paths and `..` escapes lexically.
    let relp = Path::new(rel);
    if relp.is_absolute() {
        return None;
    }
    if relp
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return None;
    }
    let full = root.join(relp);
    let meta = std::fs::metadata(&full).ok()?;
    if !meta.is_file() || meta.len() > MAX_INLINE_BYTES {
        return None;
    }
    let bytes = std::fs::read(&full).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// Truncate a long candidate path for the picker row, keeping the tail (the
/// basename is the most informative). PURE.
#[allow(dead_code)]
pub fn truncate_path(path: &str, max: usize) -> String {
    let max = max.max(4);
    if path.graphemes(true).count() <= max {
        return path.to_string();
    }
    let tail: String = path
        .graphemes(true)
        .rev()
        .take(max - 1)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("…{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_query_detects_token_under_caret() {
        // Right after typing `@sr`.
        let q = at_query("see @sr", 7).unwrap();
        assert_eq!(q.start, 4);
        assert_eq!(q.end, 7);
        assert_eq!(q.partial, "sr");

        // Bare `@` is a query with an empty partial.
        let q = at_query("@", 1).unwrap();
        assert_eq!(q.partial, "");

        // A space closes the token → no active query.
        assert!(at_query("see @src/main.rs ", 17).is_none());
        // An email-like `a@b` is NOT a query (no whitespace before '@').
        assert!(at_query("user@host", 9).is_none());
        // No '@' at all.
        assert!(at_query("plain text", 5).is_none());
    }

    #[test]
    fn apply_pick_splices_path_and_closes_token() {
        let text = "open @ma";
        let q = at_query(text, text.len()).unwrap();
        let res = apply_pick(text, &q, "src/main.rs");
        assert_eq!(res.text, "open @src/main.rs ");
        assert_eq!(res.caret, res.text.len());

        // Picking when there's trailing text keeps it.
        let text = "diff @ab end";
        let q = at_query(text, 8).unwrap(); // caret after "@ab"
        let res = apply_pick(text, &q, "a/b.rs");
        assert_eq!(res.text, "diff @a/b.rs  end");
    }

    #[test]
    fn rank_files_prefers_basename_prefix() {
        let files = vec![
            "src/main.rs".to_string(),
            "src/app/main_loop.rs".to_string(),
            "docs/readme.md".to_string(),
            "src/markdown/render.rs".to_string(),
        ];
        let ranked = rank_files("main", &files);
        // Basename prefix "main.rs" and "main_loop.rs" rank above the substring.
        assert_eq!(ranked[0], "src/main.rs");
        assert!(ranked.contains(&"src/app/main_loop.rs".to_string()));
        // A query matching nothing yields no rows.
        assert!(rank_files("zzz", &files).is_empty());
        // Empty query keeps (shortest-first) order.
        let all = rank_files("", &files);
        assert_eq!(all[0], "src/main.rs"); // shortest
        // Subsequence fallback: "smr" hits "src/markdown/render.rs".
        let sub = rank_files("smr", &files);
        assert!(sub.iter().any(|f| f.contains("render.rs")));
    }

    #[test]
    fn find_at_tokens_scans_in_root_paths() {
        let toks = find_at_tokens("look at @src/a.rs and @docs/b.md please");
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0].2, "src/a.rs");
        assert_eq!(toks[1].2, "docs/b.md");
        // `a@b` (no whitespace before @) is not a token.
        assert!(find_at_tokens("email a@b.com").is_empty());
    }

    #[test]
    fn format_inline_wraps_in_file_markers() {
        assert_eq!(
            format_inline("src/x.rs", "fn main() {}"),
            "[File: src/x.rs]\nfn main() {}\n[/File]"
        );
    }

    #[test]
    fn expand_file_refs_reads_real_file_and_skips_missing() {
        // Build a temp tree.
        let dir = std::env::temp_dir().join(format!("tui_v4_atpath_{}", std::process::id()));
        let _ = std::fs::create_dir_all(dir.join("sub"));
        std::fs::write(dir.join("sub").join("hi.txt"), "hello @file").unwrap();

        let res = expand_file_refs("see @sub/hi.txt and @missing.txt", &dir);
        assert!(res.text.contains("[File: sub/hi.txt]"));
        assert!(res.text.contains("hello @file"));
        assert!(res.text.contains("[/File]"));
        // The missing token is left verbatim, never silently dropped.
        assert!(res.text.contains("@missing.txt"));
        assert_eq!(res.inlined, vec!["sub/hi.txt".to_string()]);

        // A `..` escape is refused (left verbatim).
        let res2 = expand_file_refs("@../etc/passwd", &dir);
        assert!(res2.inlined.is_empty());
        assert!(res2.text.contains("@../etc/passwd"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
