//! input/paths.rs — gitignore-aware project file listing for the `@path` picker
//! (checklist §4: "gitignore-aware path completion"). We walk the project root
//! breadth-first, skipping anything an ignore rule excludes, and return relative
//! POSIX-style paths.
//!
//! We do NOT pull in the heavyweight `ignore` crate; a small hand-rolled matcher
//! over `.gitignore` lines covers the common cases (bare names, `dir/`, leading
//! `*`/trailing `*` globs, anchored `/path`) plus a baked-in default skip set
//! (`.git`, `target`, `node_modules`, …). The **matcher is PURE + unit-tested**;
//! the walk is a thin effectful wrapper around it. A walk failure degrades to an
//! empty list (the picker just shows nothing) — never a panic.

use std::path::Path;

/// Directories we always skip regardless of `.gitignore` (huge / irrelevant to a
/// chat `@` inline). Mirrors the spirit of a default ignore.
pub const DEFAULT_SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "target",
    // This repo's giant log/scratch tree (model_responses dumps, clones); walking it
    // dominated the `@`-picker walk cost (Q12). Skipped alongside build artifacts.
    "temp",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    ".idea",
    ".vscode",
];

/// Cap on the number of files the picker indexes (keeps the walk bounded on a
/// huge monorepo; the picker only needs enough to fuzzy-match). The walk is a
/// deterministic BFS (see [`list_project_files`]) so the cap drops only the
/// DEEPEST/last-sorted files, never a near-root source the user is likely to `@`.
pub const MAX_INDEXED_FILES: usize = 20000;

/// A compiled set of ignore rules (from one or more `.gitignore` files + the
/// default skip set). PURE matching via [`IgnoreRules::is_ignored`].
#[derive(Debug, Default, Clone)]
pub struct IgnoreRules {
    rules: Vec<Rule>,
}

#[derive(Debug, Clone)]
struct Rule {
    /// The pattern body (without a trailing `/` or leading `!`).
    pat: String,
    /// `dir/` form — matches directories only.
    dir_only: bool,
    /// Anchored to the root (had a leading `/`, or contains a non-trailing `/`).
    anchored: bool,
    /// A negation (`!pat`) — un-ignores a previously-ignored path.
    negated: bool,
}

impl IgnoreRules {
    /// Start from the default skip set (each as a bare directory rule).
    pub fn with_defaults() -> Self {
        let mut rules = Vec::new();
        for d in DEFAULT_SKIP_DIRS {
            rules.push(Rule {
                pat: (*d).to_string(),
                dir_only: false, // skip both a dir and a same-named file.
                anchored: false,
                negated: false,
            });
        }
        IgnoreRules { rules }
    }

    /// Parse `.gitignore` text and append its rules (later files override via
    /// negations). PURE.
    pub fn add_gitignore(&mut self, text: &str) {
        for raw in text.lines() {
            let line = raw.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let negated = line.starts_with('!');
            let mut body = if negated { &line[1..] } else { line };
            let dir_only = body.ends_with('/');
            if dir_only {
                body = &body[..body.len() - 1];
            }
            let leading_slash = body.starts_with('/');
            if leading_slash {
                body = &body[1..];
            }
            if body.is_empty() {
                continue;
            }
            let anchored = leading_slash || body.contains('/');
            self.rules.push(Rule {
                pat: body.to_string(),
                dir_only,
                anchored,
                negated,
            });
        }
    }

    /// True if a relative `path` (POSIX `/`-separated, e.g. `src/main.rs`) is
    /// ignored. `is_dir` lets a `dir/` rule apply only to directories — BUT a
    /// `dir/` rule that matches an ANCESTOR directory of `path` also ignores the
    /// file beneath it (gitignore: `build/` ignores `build/out.js`). Later rules
    /// win (so a trailing `!negation` can re-include). PURE.
    pub fn is_ignored(&self, path: &str, is_dir: bool) -> bool {
        let mut ignored = false;
        for r in &self.rules {
            let matched = if r.dir_only {
                // A dir-only rule matches the dir itself (when is_dir) OR any
                // ancestor directory segment of the path.
                (is_dir && rule_matches(r, path, true)) || ancestor_dir_matches(r, path)
            } else {
                rule_matches(r, path, is_dir)
            };
            if matched {
                ignored = !r.negated;
            }
        }
        ignored
    }
}

/// Whether a single rule matches `path`. Handles: anchored full-path match,
/// any-segment basename match, and `*`-glob (leading/trailing/`*.ext`). PURE.
fn rule_matches(r: &Rule, path: &str, _is_dir: bool) -> bool {
    if r.anchored {
        // Anchored: match the path from the root (prefix on a segment boundary).
        return glob_match(&r.pat, path)
            || path
                .strip_prefix(&format!("{}/", r.pat.trim_end_matches('/')))
                .is_some()
            || glob_prefix_dir(&r.pat, path);
    }
    // Unanchored: match against ANY path segment (basename semantics).
    path.split('/').any(|seg| glob_match(&r.pat, seg))
        // …and also against the full path for slash-containing globs.
        || glob_match(&r.pat, path)
}

/// True if an anchored directory pattern is a parent of `path`
/// (e.g. pattern `build` should ignore `build/x/y`).
fn glob_prefix_dir(pat: &str, path: &str) -> bool {
    if let Some(rest) = path.strip_prefix(pat) {
        return rest.starts_with('/');
    }
    false
}

/// True if a dir-only rule matches some ANCESTOR directory of `path`. For an
/// unanchored rule, any path SEGMENT before the last counts; for an anchored
/// rule, the leading directory components must match. PURE.
fn ancestor_dir_matches(r: &Rule, path: &str) -> bool {
    let segs: Vec<&str> = path.split('/').collect();
    if segs.len() <= 1 {
        return false; // no ancestor directory.
    }
    if r.anchored {
        // Anchored: rebuild the path up to each ancestor and test the rule.
        for i in 1..segs.len() {
            let ancestor = segs[..i].join("/");
            if rule_matches(r, &ancestor, true) {
                return true;
            }
        }
        false
    } else {
        // Unanchored: any ancestor SEGMENT (not the leaf) matching the glob.
        segs[..segs.len() - 1]
            .iter()
            .any(|seg| glob_match(&r.pat, seg))
    }
}

/// A tiny glob matcher supporting `*` (any run, no `/`) and `?` (one char). No
/// `**`; that's beyond what a chat `@` picker needs. PURE + unit-tested.
fn glob_match(pat: &str, text: &str) -> bool {
    fn helper(p: &[u8], t: &[u8]) -> bool {
        match p.first() {
            None => t.is_empty(),
            Some(b'*') => {
                // `*` matches zero+ chars that are not '/'.
                if helper(&p[1..], t) {
                    return true;
                }
                if let Some(&c) = t.first() {
                    if c != b'/' {
                        return helper(p, &t[1..]);
                    }
                }
                false
            }
            Some(b'?') => {
                matches!(t.first(), Some(&c) if c != b'/') && helper(&p[1..], &t[1..])
            }
            Some(&pc) => {
                matches!(t.first(), Some(&c) if c == pc) && helper(&p[1..], &t[1..])
            }
        }
    }
    helper(pat.as_bytes(), text.as_bytes())
}

/// List project files under `root` (gitignore-aware), relative POSIX paths,
/// bounded by [`MAX_INDEXED_FILES`]. Reads `root/.gitignore` if present. A walk
/// error degrades to whatever was collected so far (never a panic). Effectful.
///
/// The frontier is a FIFO `VecDeque` used as a **breadth-first** queue, and each
/// directory's entries are SORTED before enqueueing. So the walk visits files in a
/// deterministic, shallow-first / alphabetical order independent of `read_dir`'s OS
/// order — the [`MAX_INDEXED_FILES`] cap therefore drops only the DEEPEST,
/// last-sorted files, never a near-root source the user is likely to `@`-mention
/// (the old LIFO `Vec::pop()` DFS sorted only AFTER the cap, so it indexed an
/// arbitrary partial slice on a big repo).
pub fn list_project_files(root: &Path) -> Vec<String> {
    walk_with_cap(root, MAX_INDEXED_FILES)
}

/// The BFS walk with an explicit `cap` (so the cap behavior is testable with a tiny
/// bound instead of materializing [`MAX_INDEXED_FILES`] files). See
/// [`list_project_files`] for the determinism / shallow-first contract.
fn walk_with_cap(root: &Path, cap: usize) -> Vec<String> {
    let mut rules = IgnoreRules::with_defaults();
    if let Ok(text) = std::fs::read_to_string(root.join(".gitignore")) {
        rules.add_gitignore(&text);
    }
    let mut out: Vec<String> = Vec::new();
    let mut queue: std::collections::VecDeque<(std::path::PathBuf, String)> =
        std::collections::VecDeque::from([(root.to_path_buf(), String::new())]);
    while let Some((dir, prefix)) = queue.pop_front() {
        if out.len() >= cap {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        // Sort this directory's children by name so the BFS frontier is processed
        // in a deterministic order regardless of the OS read_dir order.
        let mut children: Vec<(std::path::PathBuf, String, bool)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let rel = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if rules.is_ignored(&rel, is_dir) {
                continue;
            }
            children.push((entry.path(), rel, is_dir));
        }
        children.sort_by(|a, b| a.1.cmp(&b.1));
        for (path, rel, is_dir) in children {
            if is_dir {
                queue.push_back((path, rel));
            } else {
                out.push(rel);
                if out.len() >= cap {
                    break;
                }
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_stars_and_literals() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "main.py"));
        assert!(glob_match("foo*", "foobar"));
        assert!(glob_match("*bar", "foobar"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
        assert!(glob_match("exact", "exact"));
        // `*` does not cross a '/'.
        assert!(!glob_match("*", "a/b"));
    }

    #[test]
    fn default_skips_cover_git_and_target() {
        let r = IgnoreRules::with_defaults();
        assert!(r.is_ignored(".git", true));
        assert!(r.is_ignored("target", true));
        assert!(r.is_ignored("target/debug/foo", false)); // nested under target.
        assert!(r.is_ignored("node_modules/x/y.js", false));
        assert!(!r.is_ignored("src/main.rs", false));
    }

    #[test]
    fn default_skip_excludes_temp() {
        // This repo's giant `temp/` log tree dominated the @-picker walk; it must be
        // skipped by the default ignore set (both the dir itself and anything nested
        // under it), same as `target` (Q12 @ speed).
        let r = IgnoreRules::with_defaults();
        assert!(r.is_ignored("temp", true));
        assert!(r.is_ignored("temp/model_responses/dump_42.txt", false));
        // A non-temp sibling must still be listable (no over-broad match).
        assert!(!r.is_ignored("template.rs", false));
        assert!(!r.is_ignored("src/temptation.rs", false));
    }

    #[test]
    fn gitignore_rules_apply_with_anchoring_and_negation() {
        let mut r = IgnoreRules::with_defaults();
        r.add_gitignore(
            "# comment\n\
             *.log\n\
             build/\n\
             /secrets.txt\n\
             docs/private\n\
             !docs/private/keep.md\n",
        );
        // Bare glob applies to any segment.
        assert!(r.is_ignored("a/b/c.log", false));
        // `build/` applies to a dir (and nested).
        assert!(r.is_ignored("build", true));
        assert!(r.is_ignored("build/out.js", false));
        // But a FILE named build is NOT ignored by the dir-only rule.
        assert!(!r.is_ignored("build", false));
        // Anchored `/secrets.txt` only at root.
        assert!(r.is_ignored("secrets.txt", false));
        assert!(!r.is_ignored("sub/secrets.txt", false));
        // Anchored dir path.
        assert!(r.is_ignored("docs/private/passwd", false));
        // Negation re-includes a specific file.
        assert!(!r.is_ignored("docs/private/keep.md", false));
    }

    #[test]
    fn list_project_files_walks_and_respects_ignore() {
        let dir = std::env::temp_dir().join(format!("tui_v4_walk_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join("target/debug")).unwrap();
        std::fs::write(dir.join(".gitignore"), "*.log\nsecret.txt\n").unwrap();
        std::fs::write(dir.join("src").join("main.rs"), "fn main(){}").unwrap();
        std::fs::write(dir.join("a.log"), "noise").unwrap();
        std::fs::write(dir.join("secret.txt"), "x").unwrap();
        std::fs::write(dir.join("target").join("debug").join("big.o"), "obj").unwrap();
        std::fs::write(dir.join("readme.md"), "hi").unwrap();

        let files = list_project_files(&dir);
        assert!(files.contains(&"src/main.rs".to_string()));
        assert!(files.contains(&"readme.md".to_string()));
        // Ignored by .gitignore.
        assert!(!files.iter().any(|f| f.ends_with(".log")));
        assert!(!files.contains(&"secret.txt".to_string()));
        // Ignored by the default skip set (target/).
        assert!(!files.iter().any(|f| f.starts_with("target/")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn walk_is_deterministic_and_cap_drops_deepest_first() {
        // A tree with many SHALLOW (root-level) files plus one DEEP file. With the
        // deterministic BFS, the shallow files are visited first, so a small cap
        // drops the deep file — never a near-root source the user is likely to `@`.
        let dir = std::env::temp_dir().join(format!("tui_v4_bfs_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("deep/a/b/c")).unwrap();
        for i in 0..8 {
            std::fs::write(dir.join(format!("root_{i:02}.rs")), "x").unwrap();
        }
        std::fs::write(dir.join("deep/a/b/c/leaf.rs"), "deep").unwrap();

        // Determinism: two independent builds yield byte-identical ordering.
        let a = list_project_files(&dir);
        let b = list_project_files(&dir);
        assert_eq!(a, b, "BFS walk order must be deterministic across builds");

        // A known shallow file is always present under the cap; cap < shallow count
        // drops the DEEP leaf, not the shallow roots.
        let capped = walk_with_cap(&dir, 5);
        assert_eq!(capped.len(), 5);
        assert!(capped.contains(&"root_00.rs".to_string()), "shallow file kept under cap");
        assert!(
            !capped.iter().any(|f| f.contains("leaf.rs")),
            "deepest file dropped first under the cap, not a shallow one"
        );
        // Two capped builds are also identical (cap doesn't introduce nondeterminism).
        assert_eq!(walk_with_cap(&dir, 5), capped);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
