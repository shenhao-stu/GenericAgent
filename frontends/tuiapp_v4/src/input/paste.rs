//! input/paste.rs — paste folding + placeholder-aware deletion (checklist §4
//! "placeholder-aware backspace"; §8 "Placeholder-aware backspace … whole-block";
//! tui_v3 §D 49-55). A big / multi-line paste folds to a `[Pasted text #N]`
//! placeholder; a clipboard image path to `[Image #N]`; a file path to
//! `[File #N]`. The real payload is held in a side-table keyed `#N`. A Backspace
//! flush against the END of a placeholder (or Delete against its START) wipes the
//! WHOLE token and drops its store entry — never a lone bracket.
//!
//! On submit the placeholders expand back to their payloads (text inline; image
//! paths collected as `Submit.images`). All PURE + unit-tested.

use std::collections::HashMap;

/// What a stored placeholder stands for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PasteKind {
    /// A folded multi-line / large text paste — expands back to the text inline.
    Text(String),
    /// A pasted image path — collected as a `Submit.images[]` entry, not inlined.
    Image(String),
    /// A pasted file path — expands to the path (so `@`/file expansion can run).
    File(String),
}

/// The side-table of folded paste payloads, keyed by the integer `#N`.
#[derive(Debug, Default, Clone)]
pub struct PasteStore {
    next: u32,
    entries: HashMap<u32, PasteKind>,
}

/// The result of folding a paste: the placeholder to INSERT into the buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldResult {
    /// The `[Pasted text #N]` / `[Image #N]` / `[File #N]` token.
    pub insert: String,
    /// The id `N` assigned (for tests / audit).
    pub id: u32,
}

impl PasteStore {
    pub fn new() -> Self {
        PasteStore {
            next: 1,
            entries: HashMap::new(),
        }
    }

    fn alloc(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }

    /// Fold a (large / multi-line) text paste → `[Pasted text #N +K lines]`.
    pub fn fold_text(&mut self, text: &str) -> FoldResult {
        let id = self.alloc();
        let lines = text.matches('\n').count();
        self.entries.insert(id, PasteKind::Text(text.to_string()));
        let insert = if lines > 0 {
            format!("[Pasted text #{id} +{lines} lines]")
        } else {
            format!("[Pasted text #{id}]")
        };
        FoldResult { insert, id }
    }

    /// Fold a pasted image path → `[Image #N]`.
    pub fn fold_image(&mut self, path: &str) -> FoldResult {
        let id = self.alloc();
        self.entries.insert(id, PasteKind::Image(path.to_string()));
        FoldResult {
            insert: format!("[Image #{id}]"),
            id,
        }
    }

    /// Fold a pasted file path → `[File #N]`.
    pub fn fold_file(&mut self, path: &str) -> FoldResult {
        let id = self.alloc();
        self.entries.insert(id, PasteKind::File(path.to_string()));
        FoldResult {
            insert: format!("[File #{id}]"),
            id,
        }
    }

    /// Look up a stored payload by id.
    #[allow(dead_code)] // store inspection (tested; used by the Phase-3 image path).
    pub fn get(&self, id: u32) -> Option<&PasteKind> {
        self.entries.get(&id)
    }

    /// Drop a stored payload (called when its placeholder is deleted).
    pub fn remove(&mut self, id: u32) {
        self.entries.remove(&id);
    }

    /// Expand every placeholder in `text` back to its payload (text inline, file
    /// path inline; an image placeholder is REMOVED from text since its path
    /// travels via [`collect_images`]). PURE over the store.
    pub fn expand(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for ph in find_placeholders(text) {
            out.push_str(&text[cursor..ph.start]);
            match self.entries.get(&ph.id) {
                Some(PasteKind::Text(t)) => out.push_str(t),
                Some(PasteKind::File(p)) => out.push_str(p),
                Some(PasteKind::Image(_)) | None => { /* image → dropped from text */ }
            }
            cursor = ph.end;
        }
        out.push_str(&text[cursor..]);
        out
    }

    /// Collect the image paths referenced by `[Image #N]` placeholders present in
    /// `text`, in order (the `Submit.images` payload).
    #[allow(dead_code)] // image attachment wires through here in Phase 3.
    pub fn collect_images(&self, text: &str) -> Vec<String> {
        let mut imgs = Vec::new();
        for ph in find_placeholders(text) {
            if let Some(PasteKind::Image(p)) = self.entries.get(&ph.id) {
                imgs.push(p.clone());
            }
        }
        imgs
    }
}

/// A placeholder token located in the buffer: its byte span and id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placeholder {
    pub start: usize,
    pub end: usize,
    pub id: u32,
}

/// Find every `[Pasted text #N …]` / `[Image #N]` / `[File #N]` token in `text`.
/// PURE — a simple scan for the three bracketed forms with a trailing `#<digits>`.
pub fn find_placeholders(text: &str) -> Vec<Placeholder> {
    const PREFIXES: &[&str] = &["[Pasted text #", "[Image #", "[File #"];
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    'scan: while i < bytes.len() {
        if bytes[i] == b'[' {
            for pre in PREFIXES {
                if text[i..].starts_with(pre) {
                    // Parse digits after the prefix.
                    let num_start = i + pre.len();
                    let mut j = num_start;
                    while j < bytes.len() && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > num_start {
                        // Find the closing ']'.
                        if let Some(close_rel) = text[j..].find(']') {
                            let end = j + close_rel + 1;
                            if let Ok(id) = text[num_start..j].parse::<u32>() {
                                out.push(Placeholder { start: i, end, id });
                                i = end;
                                continue 'scan;
                            }
                        }
                    }
                }
            }
        }
        i += 1;
    }
    out
}

/// The placeholder ADJACENT to `caret` on the given side, if any. `Left` = the
/// caret is at the END of a placeholder (Backspace target); `Right` = the caret
/// is at the START of a placeholder (Delete target). PURE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// Whether a placeholder is flush against `caret` on `side` (whole-block delete
/// target). Returns the placeholder span+id to wipe.
pub fn placeholder_adjacent(text: &str, caret: usize, side: Side) -> Option<Placeholder> {
    let phs = find_placeholders(text);
    match side {
        Side::Left => phs.into_iter().find(|p| p.end == caret),
        Side::Right => phs.into_iter().find(|p| p.start == caret),
    }
}

/// Delete a whole placeholder from `text`, dropping its store entry. Returns the
/// new text + the caret position (at the deletion point). PURE over text; mutates
/// the store (removes the id).
pub fn delete_placeholder(text: &str, ph: Placeholder, store: &mut PasteStore) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..ph.start]);
    out.push_str(&text[ph.end..]);
    store.remove(ph.id);
    (out, ph.start)
}

/// Heuristic: does a single-line pasted string look like an on-disk path
/// (tui_v3 `_try_paste_path`)? Conservative so an ordinary one-line message is
/// NOT mistaken for a file. PURE.
pub fn looks_like_path(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() || s.contains("  ") {
        return false; // double-space run → prose.
    }
    let has_sep = s.contains('/') || s.contains('\\');
    let has_drive = {
        let b = s.as_bytes();
        b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/')
    };
    let has_ext = s
        .rsplit('.')
        .next()
        .map(|ext| !ext.is_empty() && ext.len() <= 8 && ext.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or(false)
        && s.contains('.');
    (has_drive || has_sep) && has_ext
}

/// Is `s` an image path (by extension)? Drives image vs file classification of a
/// pasted path. PURE.
pub fn is_image_path(s: &str) -> bool {
    const EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"];
    let lower = s.to_ascii_lowercase();
    EXTS.iter().any(|e| lower.ends_with(&format!(".{e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_text_image_file_assign_ids_and_tokens() {
        let mut s = PasteStore::new();
        let t = s.fold_text("a\nb\nc");
        assert_eq!(t.insert, "[Pasted text #1 +2 lines]");
        let img = s.fold_image("/tmp/shot.png");
        assert_eq!(img.insert, "[Image #2]");
        let f = s.fold_file("src/main.rs");
        assert_eq!(f.insert, "[File #3]");
        // Single-line text paste has no "+K lines" suffix.
        let one = s.fold_text("just one line");
        assert_eq!(one.insert, "[Pasted text #4]");
    }

    #[test]
    fn find_placeholders_locates_all_three_forms() {
        let text = "see [Pasted text #1 +3 lines] and [Image #2] then [File #3] end";
        let phs = find_placeholders(text);
        assert_eq!(phs.len(), 3);
        assert_eq!(phs[0].id, 1);
        assert_eq!(phs[1].id, 2);
        assert_eq!(phs[2].id, 3);
        // The spans really cover the bracketed tokens.
        assert_eq!(&text[phs[1].start..phs[1].end], "[Image #2]");
    }

    #[test]
    fn placeholder_aware_whole_block_delete() {
        let mut s = PasteStore::new();
        let r = s.fold_text("x\ny");
        let text = format!("hi {} bye", r.insert); // "hi [Pasted text #1 +1 lines] bye"
        // Caret at the END of the placeholder → Backspace wipes the whole token.
        let end = text.find(" bye").unwrap();
        let ph = placeholder_adjacent(&text, end, Side::Left).expect("left-adjacent");
        assert_eq!(ph.id, 1);
        let (out, caret) = delete_placeholder(&text, ph, &mut s);
        assert_eq!(out, "hi  bye");
        assert_eq!(caret, 3);
        // The store entry is dropped.
        assert!(s.get(1).is_none());

        // Caret at the START → Delete wipes it (Right side).
        let mut s2 = PasteStore::new();
        let r2 = s2.fold_image("/a/b.png");
        let text2 = format!("{} tail", r2.insert);
        let ph2 = placeholder_adjacent(&text2, 0, Side::Right).expect("right-adjacent");
        let (out2, caret2) = delete_placeholder(&text2, ph2, &mut s2);
        assert_eq!(out2, " tail");
        assert_eq!(caret2, 0);
    }

    #[test]
    fn expand_inlines_text_and_file_but_routes_image_to_images() {
        let mut s = PasteStore::new();
        let t = s.fold_text("hello\nworld");
        let img = s.fold_image("/tmp/a.png");
        let f = s.fold_file("src/x.rs");
        let buf = format!("{} {} {}", t.insert, img.insert, f.insert);
        // Text + file inline; image dropped from text.
        let expanded = s.expand(&buf);
        assert!(expanded.contains("hello\nworld"));
        assert!(expanded.contains("src/x.rs"));
        assert!(!expanded.contains("a.png"));
        // The image path is collected for Submit.images.
        assert_eq!(s.collect_images(&buf), vec!["/tmp/a.png".to_string()]);
    }

    #[test]
    fn path_heuristics() {
        assert!(looks_like_path("src/main.rs"));
        assert!(looks_like_path("C:\\Users\\me\\shot.png"));
        assert!(!looks_like_path("just a normal sentence here"));
        assert!(!looks_like_path("hello world")); // no sep/ext.
        assert!(is_image_path("/tmp/screenshot.PNG"));
        assert!(!is_image_path("notes.txt"));
    }
}
