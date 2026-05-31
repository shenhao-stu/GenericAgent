//! markdown/math.rs — `latex_to_unicode`: render LaTeX math (`$…$` inline,
//! `$$…$$` block) into terminal-friendly Unicode (checklist §1 P3, §3
//! markdown/math). Claude Code has NO math rendering — this is the surpass-CC
//! feature.
//!
//! The terminal is a fixed-width character grid, so a full LaTeX engine's 2-D
//! box layout would be thrown away and re-laid-out for cells anyway. Instead
//! (per recon `painpoints/markdown_math.md` §5/§6, the authoritative spec for
//! this slice) we use a **curated symbol map + small structural transforms**:
//!
//!   * Greek letters, operators, relations, arrows → Unicode glyphs.
//!   * `^`/`_` → Unicode super/subscripts where the glyph exists, else `^(…)` /
//!     `_(…)` (NEVER a wrong/blank glyph).
//!   * `\frac{a}{b}` → `a/b` inline, a 3-line stacked `num / ─── / den` for block.
//!   * `\sqrt{x}` → `√(x)` inline, an overbar + `√ x` for block; `\sqrt[n]{x}`
//!     keeps the index.
//!   * `\sum`/`\prod`/`\int` with `_lo^hi` limits → glyph+sub/sup inline, a
//!     3-line stack for block.
//!   * `\hat`/`\bar`/`\vec`/`\dot`/`\tilde` → combining accent on the base.
//!   * `\begin{pmatrix}…\end{…}` → bracketed grid (inline `(a b; c d)`).
//!
//! **Hard contract (P3):** the pipeline is pure string→string, is allocation-
//! bounded, and on ANY parse failure returns the *original* LaTeX verbatim — it
//! must NEVER panic and never corrupt the surrounding stream. All width math
//! uses display cells (`unicode-width`), never byte/char length, so CJK and
//! combining marks align.

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Whether math renders on one line (`$…$`) or may stack into several rows
/// (`$$…$$`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Display {
    /// `$…$` — collapse to a single line (`\frac{a}{b}` → `a/b`).
    Inline,
    /// `$$…$$` — may produce a multi-line stacked layout (3-line `\frac`, etc.).
    Block,
}

/// The result of rendering one math expression. `lines.len() == 1` for inline;
/// block math may stack (e.g. a fraction is 3 rows). `baseline` is the index of
/// the "main" row, used to vertically center this fragment when it is stacked
/// side-by-side with another (e.g. `\sum` limits beside its operand).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathRender {
    /// The rendered rows (display strings, each free of `\n`).
    pub lines: Vec<String>,
    /// Max display width across `lines` (terminal cells).
    pub width: usize,
    /// Index into `lines` of the baseline row (for side-by-side vertical align).
    pub baseline: usize,
}

impl MathRender {
    /// A single-line fragment (the common inline case).
    fn single(s: String) -> MathRender {
        let width = UnicodeWidthStr::width(s.as_str());
        MathRender {
            lines: vec![s],
            width,
            baseline: 0,
        }
    }

    /// A multi-line fragment with an explicit baseline.
    fn multi(lines: Vec<String>, baseline: usize) -> MathRender {
        let width = lines
            .iter()
            .map(|l| UnicodeWidthStr::width(l.as_str()))
            .max()
            .unwrap_or(0);
        MathRender {
            lines,
            width,
            baseline,
        }
    }
}

/// Render LaTeX math to Unicode. The PUBLIC entry point (P3).
///
/// `latex` is the math body **without** the surrounding `$`/`$$` delimiters.
/// On any internal failure this returns the input verbatim (one line) — the
/// "never panic, never corrupt the stream" guarantee. It is wrapped in
/// `catch_unwind` as a belt-and-braces final backstop even though the transforms
/// are written to be total.
pub fn latex_to_unicode(latex: &str, display: Display) -> MathRender {
    let input = latex.trim();
    if input.is_empty() {
        return MathRender::single(String::new());
    }
    // Belt-and-braces: even a logic bug below degrades to the literal LaTeX
    // rather than taking down the render. The transforms themselves never panic
    // on valid UTF-8, but a future edit shouldn't be able to break the stream.
    let rendered = std::panic::catch_unwind(|| {
        let tokens = tokenize(input);
        render_tokens(&tokens, display)
    });
    match rendered {
        Ok(r) => r,
        Err(_) => MathRender::single(input.to_string()),
    }
}

/// Convenience: render to a single `\n`-joined string (for callers that just
/// want text). Inline always yields one line; block may contain `\n`.
#[allow(dead_code)] // used by the markdown emitter + available to fwd callers.
pub fn render_math(latex: &str, display: Display) -> String {
    latex_to_unicode(latex, display).lines.join("\n")
}

// ===========================================================================
// Tokenizer
// ===========================================================================

/// A flat LaTeX token. `^`/`_` attach to the FOLLOWING atom at render time.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    /// A `\command` (backslash stripped). e.g. `alpha`, `frac`, `sqrt`, `begin`.
    Cmd(String),
    /// A balanced `{…}` group, tokenized recursively.
    Group(Vec<Tok>),
    /// A bracketed `[…]` optional argument (for `\sqrt[n]{…}`), raw text.
    Opt(String),
    /// A single source character (letters, digits, operators, spaces).
    Char(char),
    /// `^` — superscript marker (applies to the next atom).
    Sup,
    /// `_` — subscript marker (applies to the next atom).
    Sub,
    /// `&` — matrix column separator.
    Amp,
    /// `\\` — matrix row break.
    RowBreak,
}

/// Tokenize a LaTeX string into a flat `Vec<Tok>` (groups nested recursively).
/// Pure + total over valid UTF-8.
fn tokenize(s: &str) -> Vec<Tok> {
    let chars: Vec<char> = s.chars().collect();
    let (toks, _) = tokenize_from(&chars, 0, false);
    toks
}

/// Tokenize starting at `i`. When `in_group` is true, stop at the matching `}`
/// and return the index just past it; otherwise consume to the end. Returns the
/// tokens and the next index.
fn tokenize_from(chars: &[char], mut i: usize, in_group: bool) -> (Vec<Tok>, usize) {
    let mut out: Vec<Tok> = Vec::new();
    while i < chars.len() {
        let c = chars[i];
        match c {
            '}' if in_group => {
                return (out, i + 1);
            }
            '\\' => {
                // Escaped row break `\\`, escaped delimiter `\$`/`\{`/`\}`, or a
                // `\command` (alphabetic run). A lone trailing backslash is kept
                // literal.
                if i + 1 < chars.len() && chars[i + 1] == '\\' {
                    out.push(Tok::RowBreak);
                    i += 2;
                } else if i + 1 < chars.len() && !chars[i + 1].is_ascii_alphabetic() {
                    // `\$`, `\,`, `\!`, `\{`, `\}`, `\ ` … keep the command name as
                    // the single following char so the symbol map can map it
                    // (e.g. `\,` → thin space) or it falls through literally.
                    out.push(Tok::Cmd(chars[i + 1].to_string()));
                    i += 2;
                } else {
                    let mut j = i + 1;
                    while j < chars.len() && chars[j].is_ascii_alphabetic() {
                        j += 1;
                    }
                    let name: String = chars[i + 1..j].iter().collect();
                    out.push(Tok::Cmd(name));
                    i = j;
                }
            }
            '{' => {
                let (inner, next) = tokenize_from(chars, i + 1, true);
                out.push(Tok::Group(inner));
                i = next;
            }
            '[' => {
                // Capture an optional `[…]` argument as raw text (for \sqrt[n]).
                // If there is no closing `]`, treat `[` as a literal char.
                if let Some(close) = find_close_bracket(chars, i + 1) {
                    let text: String = chars[i + 1..close].iter().collect();
                    out.push(Tok::Opt(text));
                    i = close + 1;
                } else {
                    out.push(Tok::Char('['));
                    i += 1;
                }
            }
            '^' => {
                out.push(Tok::Sup);
                i += 1;
            }
            '_' => {
                out.push(Tok::Sub);
                i += 1;
            }
            '&' => {
                out.push(Tok::Amp);
                i += 1;
            }
            _ => {
                out.push(Tok::Char(c));
                i += 1;
            }
        }
    }
    (out, i)
}

/// Find the index of the `]` that closes an optional arg started just after `[`,
/// scanning at the same nesting level. `None` if unbalanced (so `[` stays raw).
fn find_close_bracket(chars: &[char], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = start;
    while i < chars.len() {
        match chars[i] {
            '[' => depth += 1,
            ']' if depth == 0 => return Some(i),
            ']' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    None
}

// ===========================================================================
// Renderer
// ===========================================================================

/// Render a flat token list. Walks atoms left→right, folding `^`/`_` onto the
/// preceding atom, dispatching structural commands (`\frac`, `\sqrt`, big-ops,
/// accents, matrices) and mapping bare symbols.
fn render_tokens(tokens: &[Tok], display: Display) -> MathRender {
    // A matrix `\begin{…}…\end{…}` consumes a whole run; detect it up front so
    // the surrounding row math doesn't try to render `begin`/`end` literally.
    if let Some(mat) = try_render_matrix(tokens, display) {
        return mat;
    }

    let mut frags: Vec<MathRender> = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        match &tokens[i] {
            Tok::Sup | Tok::Sub => {
                // A leading sup/sub with nothing before it: render its atom with
                // the marker as a literal prefix (degraded but never lost).
                let is_sup = matches!(tokens[i], Tok::Sup);
                let (atom, next) = next_atom(tokens, i + 1);
                let inner = render_atom_inline(&atom, display);
                let s = if is_sup {
                    to_superscript(&inner)
                } else {
                    to_subscript(&inner)
                };
                frags.push(MathRender::single(s));
                i = next;
            }
            Tok::Cmd(name) if is_bigop(name) => {
                // `\sum`/`\int`/… optionally followed by `_lo` and/or `^hi`.
                let glyph = symbol(name).unwrap_or_else(|| name.clone());
                let mut sub: Option<Vec<Tok>> = None;
                let mut sup: Option<Vec<Tok>> = None;
                let mut j = i + 1;
                // Limits may appear in either order: _a^b or ^b_a.
                for _ in 0..2 {
                    match tokens.get(j) {
                        Some(Tok::Sub) => {
                            let (atom, next) = next_atom(tokens, j + 1);
                            sub = Some(atom);
                            j = next;
                        }
                        Some(Tok::Sup) => {
                            let (atom, next) = next_atom(tokens, j + 1);
                            sup = Some(atom);
                            j = next;
                        }
                        _ => break,
                    }
                }
                frags.push(render_bigop(&glyph, sub.as_deref(), sup.as_deref(), display));
                i = j;
            }
            Tok::Cmd(name) if name == "frac" || name == "dfrac" || name == "tfrac" => {
                // \frac{NUM}{DEN}
                let (num, j) = next_group_tokens(tokens, i + 1);
                let (den, k) = next_group_tokens(tokens, j);
                frags.push(render_frac(&num, &den, display));
                i = k;
            }
            Tok::Cmd(name) if name == "sqrt" => {
                // \sqrt[idx]{RAD} — the index is optional.
                let mut j = i + 1;
                let mut index: Option<String> = None;
                if let Some(Tok::Opt(idx)) = tokens.get(j) {
                    index = Some(idx.clone());
                    j += 1;
                }
                let (rad, k) = next_group_tokens(tokens, j);
                frags.push(render_sqrt(&rad, index.as_deref(), display));
                i = k;
            }
            Tok::Cmd(name) if is_accent(name) => {
                // \hat{x} \bar{x} \vec{x} …
                let (base, j) = next_group_tokens(tokens, i + 1);
                frags.push(render_accent(name, &base));
                i = j;
            }
            Tok::Cmd(name) if name == "text" || name == "mathrm" || name == "operatorname" => {
                // \text{…}: render the inner literally (no symbol mapping).
                let (inner, j) = next_group_tokens(tokens, i + 1);
                let s: String = inner
                    .iter()
                    .map(|t| match t {
                        Tok::Char(c) => c.to_string(),
                        _ => String::new(),
                    })
                    .collect();
                frags.push(MathRender::single(s));
                i = j;
            }
            Tok::Cmd(name) if name == "left" || name == "right" => {
                // \left( / \right) — the delimiter follows as the next char/cmd.
                // We keep the delimiter and drop the sizing command.
                if let Some(next_tok) = tokens.get(i + 1) {
                    let d = match next_tok {
                        Tok::Char('.') => String::new(), // \left. = invisible
                        Tok::Char(c) => c.to_string(),
                        Tok::Cmd(c) => symbol(c).unwrap_or_else(|| c.clone()),
                        _ => String::new(),
                    };
                    if !d.is_empty() {
                        frags.push(MathRender::single(d));
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            Tok::Cmd(name) => {
                // A bare symbol command, possibly carrying a sup/sub after it
                // (e.g. \alpha^2). Render the glyph, then fold trailing markers.
                let glyph = match symbol(name) {
                    Some(g) => g,
                    None => {
                        // Unknown command: keep it literal with the backslash so
                        // nothing is silently dropped (P3 "return the original").
                        format!("\\{name}")
                    }
                };
                let (frag, next) = fold_scripts(MathRender::single(glyph), tokens, i + 1, display);
                frags.push(frag);
                i = next;
            }
            Tok::Group(inner) => {
                // A standalone group: render it, then fold trailing scripts.
                let g = render_tokens(inner, display);
                let (frag, next) = fold_scripts(g, tokens, i + 1, display);
                frags.push(frag);
                i = next;
            }
            Tok::Char(c) => {
                let base = MathRender::single(c.to_string());
                let (frag, next) = fold_scripts(base, tokens, i + 1, display);
                frags.push(frag);
                i = next;
            }
            Tok::Opt(text) => {
                // A stray `[…]` not consumed by \sqrt: render literally.
                frags.push(MathRender::single(format!("[{text}]")));
                i += 1;
            }
            Tok::Amp => {
                frags.push(MathRender::single(" ".to_string()));
                i += 1;
            }
            Tok::RowBreak => {
                // A stray row break outside a matrix → a space inline.
                frags.push(MathRender::single(" ".to_string()));
                i += 1;
            }
        }
    }

    concat_fragments(&frags)
}

/// If the FOLLOWING token is `^`/`_`, fold the script onto `base` and return the
/// combined fragment + the index past the consumed script(s). Handles a sup then
/// a sub (or vice versa). Pure.
fn fold_scripts(
    base: MathRender,
    tokens: &[Tok],
    mut i: usize,
    display: Display,
) -> (MathRender, usize) {
    let mut sup: Option<String> = None;
    let mut sub: Option<String> = None;
    for _ in 0..2 {
        match tokens.get(i) {
            Some(Tok::Sup) if sup.is_none() => {
                let (atom, next) = next_atom(tokens, i + 1);
                sup = Some(to_superscript(&render_atom_inline(&atom, display)));
                i = next;
            }
            Some(Tok::Sub) if sub.is_none() => {
                let (atom, next) = next_atom(tokens, i + 1);
                sub = Some(to_subscript(&render_atom_inline(&atom, display)));
                i = next;
            }
            _ => break,
        }
    }
    if sup.is_none() && sub.is_none() {
        return (base, i);
    }
    // Single-line base: append sub then sup (math convention x_i^2 → xᵢ²). If the
    // base is multi-line we still attach to its baseline row.
    let mut out = base;
    let tail = format!("{}{}", sub.unwrap_or_default(), sup.unwrap_or_default());
    if !tail.is_empty() {
        let bl = out.baseline.min(out.lines.len().saturating_sub(1));
        if let Some(line) = out.lines.get_mut(bl) {
            line.push_str(&tail);
        }
        out.width = out
            .lines
            .iter()
            .map(|l| UnicodeWidthStr::width(l.as_str()))
            .max()
            .unwrap_or(0);
    }
    (out, i)
}

/// The "next atom" after a `^`/`_`/command argument position: a single `{…}`
/// group (returned as its inner tokens) or a single char/command. Returns the
/// atom's tokens and the index past it.
fn next_atom(tokens: &[Tok], i: usize) -> (Vec<Tok>, usize) {
    match tokens.get(i) {
        Some(Tok::Group(inner)) => (inner.clone(), i + 1),
        Some(t) => (vec![t.clone()], i + 1),
        None => (Vec::new(), i),
    }
}

/// Like [`next_atom`] but specifically for a `{…}` argument; if the next token is
/// not a group (malformed LaTeX) it still takes a single atom so we degrade
/// instead of panicking.
fn next_group_tokens(tokens: &[Tok], i: usize) -> (Vec<Tok>, usize) {
    next_atom(tokens, i)
}

/// Render an atom's tokens to a single inline string (used for sup/sub contents
/// and limits, which must collapse to one line).
fn render_atom_inline(tokens: &[Tok], _display: Display) -> String {
    render_tokens(tokens, Display::Inline)
        .lines
        .first()
        .cloned()
        .unwrap_or_default()
}

/// Concatenate sibling fragments. If all are single-line, the result is one line.
/// If any is multi-line (a block frac/sqrt/bigop), the fragments are stacked
/// side-by-side aligned on their baselines (shorter ones padded vertically).
fn concat_fragments(frags: &[MathRender]) -> MathRender {
    if frags.is_empty() {
        return MathRender::single(String::new());
    }
    if frags.iter().all(|f| f.lines.len() == 1) {
        let s: String = frags.iter().map(|f| f.lines[0].as_str()).collect();
        return MathRender::single(s);
    }
    // Multi-line: align on baselines. Compute rows above/below the baseline.
    let above = frags.iter().map(|f| f.baseline).max().unwrap_or(0);
    let below = frags
        .iter()
        .map(|f| f.lines.len() - 1 - f.baseline)
        .max()
        .unwrap_or(0);
    let total = above + below + 1;
    let mut rows = vec![String::new(); total];
    for f in frags {
        let pad_top = above - f.baseline;
        for r in 0..total {
            let cell = if r >= pad_top && r < pad_top + f.lines.len() {
                f.lines[r - pad_top].clone()
            } else {
                " ".repeat(f.width)
            };
            // Right-pad the cell to the fragment width so columns stay aligned.
            let cell = pad_right(&cell, f.width);
            rows[r].push_str(&cell);
        }
    }
    MathRender::multi(rows, above)
}

// ---- structural transforms (recon §6.4–6.8) -------------------------------

/// `\frac{num}{den}`: `num/den` inline (parenthesizing a side that has a
/// top-level binary operator), a 3-line stacked layout for block.
fn render_frac(num_toks: &[Tok], den_toks: &[Tok], display: Display) -> MathRender {
    let num = render_tokens(num_toks, display);
    let den = render_tokens(den_toks, display);

    let inline = display == Display::Inline || num.lines.len() > 1 || den.lines.len() > 1;
    if inline {
        let mut ns = num.lines.first().cloned().unwrap_or_default();
        let mut ds = den.lines.first().cloned().unwrap_or_default();
        if has_top_level_binop(&ns) {
            ns = format!("({ns})");
        }
        if has_top_level_binop(&ds) {
            ds = format!("({ds})");
        }
        return MathRender::single(format!("{ns}/{ds}"));
    }

    // Block, both sides single-line → 3-line stack with a U+2500 bar.
    let ns = num.lines.first().cloned().unwrap_or_default();
    let ds = den.lines.first().cloned().unwrap_or_default();
    let w = UnicodeWidthStr::width(ns.as_str()).max(UnicodeWidthStr::width(ds.as_str()));
    let bar = "─".repeat(w);
    let top = center(&ns, w);
    let bot = center(&ds, w);
    MathRender::multi(vec![top, bar, bot], 1)
}

/// `\sqrt{rad}` / `\sqrt[idx]{rad}`: `√(rad)` (or `ⁿ√(rad)`) inline; a 2-line
/// overbar + `√ rad` for block.
fn render_sqrt(rad_toks: &[Tok], index: Option<&str>, display: Display) -> MathRender {
    let body = render_tokens(rad_toks, display);
    let inline = display == Display::Inline || body.lines.len() > 1;
    if inline {
        let inner = body.lines.first().cloned().unwrap_or_default();
        let mut s = format!("√({inner})");
        if let Some(idx) = index {
            s = format!("{}{s}", sup_index(idx));
        }
        return MathRender::single(s);
    }
    // Block: overbar above, "√ body" below.
    let inner = body.lines.first().cloned().unwrap_or_default();
    let w = UnicodeWidthStr::width(inner.as_str());
    let lead = index.map(sup_index).unwrap_or_default();
    let lead_w = UnicodeWidthStr::width(lead.as_str());
    let vinc = "‾".repeat(w + 1);
    let top = format!("{}{}", " ".repeat(lead_w + 2), vinc);
    let base = format!("{lead}√ {inner}");
    MathRender::multi(vec![top, base], 1)
}

/// `\hat{x}` `\bar{x}` `\vec{x}` `\dot{x}` `\tilde{x}` → combining accent on the
/// base. For an overline-style accent over a multi-char base, the mark is placed
/// over each grapheme so it spans (e.g. `\bar{AB}` → `A̅B̅`).
fn render_accent(name: &str, base_toks: &[Tok]) -> MathRender {
    let base = render_atom_inline(base_toks, Display::Inline);
    let mark = accent_mark(name).unwrap_or("");
    if mark.is_empty() {
        return MathRender::single(base);
    }
    let graphemes: Vec<&str> = base.graphemes(true).collect();
    let spanning = matches!(name, "bar" | "overline" | "vec" | "widehat" | "widetilde");
    let out = if graphemes.len() <= 1 {
        format!("{base}{mark}")
    } else if spanning {
        graphemes
            .iter()
            .map(|g| format!("{g}{mark}"))
            .collect::<String>()
    } else {
        // Accent on the first grapheme only (e.g. \hat over a 2-char base).
        let mut s = String::new();
        s.push_str(graphemes[0]);
        s.push_str(mark);
        for g in &graphemes[1..] {
            s.push_str(g);
        }
        s
    };
    MathRender::single(out)
}

/// `\sum`/`\prod`/`\int` with optional `_lo`/`^hi` limits: glyph with sub/sup
/// beside it inline; a 3-line stack (hi / glyph / lo) for block.
fn render_bigop(
    glyph: &str,
    sub: Option<&[Tok]>,
    sup: Option<&[Tok]>,
    display: Display,
) -> MathRender {
    let lo = sub
        .map(|t| render_atom_inline(t, Display::Inline))
        .unwrap_or_default();
    let hi = sup
        .map(|t| render_atom_inline(t, Display::Inline))
        .unwrap_or_default();
    if display == Display::Inline {
        let mut s = glyph.to_string();
        if !lo.is_empty() {
            s.push_str(&to_subscript(&lo));
        }
        if !hi.is_empty() {
            s.push_str(&to_superscript(&hi));
        }
        return MathRender::single(s);
    }
    // Block: stack the upper limit, the glyph, and the lower limit, centered.
    let w = [
        UnicodeWidthStr::width(hi.as_str()),
        UnicodeWidthStr::width(lo.as_str()),
        UnicodeWidthStr::width(glyph),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let lines = vec![center(&hi, w), center(glyph, w), center(&lo, w)];
    MathRender::multi(lines, 1)
}

/// Detect and render a `\begin{ENV}…\end{ENV}` matrix anywhere in `tokens`.
/// Returns `None` if there is no matrix environment (so normal rendering runs).
fn try_render_matrix(tokens: &[Tok], display: Display) -> Option<MathRender> {
    // Find `\begin` `{env}` … `\end` `{env}`.
    let begin = tokens.iter().position(|t| matches!(t, Tok::Cmd(c) if c == "begin"))?;
    let env = match tokens.get(begin + 1) {
        Some(Tok::Group(inner)) => group_text(inner),
        _ => return None,
    };
    if !is_matrix_env(&env) {
        return None;
    }
    let end = tokens
        .iter()
        .position(|t| matches!(t, Tok::Cmd(c) if c == "end"))?;
    if end <= begin + 1 {
        return None;
    }
    let body = &tokens[begin + 2..end];

    // Split into rows on RowBreak, cells on Amp.
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut cur_row: Vec<Vec<Tok>> = vec![Vec::new()];
    for t in body {
        match t {
            Tok::RowBreak => {
                let rendered: Vec<String> = cur_row
                    .iter()
                    .map(|cell| render_atom_inline(cell, display))
                    .collect();
                rows.push(rendered);
                cur_row = vec![Vec::new()];
            }
            Tok::Amp => cur_row.push(Vec::new()),
            other => cur_row.last_mut().unwrap().push(other.clone()),
        }
    }
    // Final row (if non-empty / or matrix had a single row).
    if cur_row.iter().any(|c| !c.is_empty()) || rows.is_empty() {
        let rendered: Vec<String> = cur_row
            .iter()
            .map(|cell| render_atom_inline(cell, display))
            .collect();
        rows.push(rendered);
    }

    // Per-column widths.
    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut col_w = vec![0usize; ncols];
    for r in &rows {
        for (i, c) in r.iter().enumerate() {
            col_w[i] = col_w[i].max(UnicodeWidthStr::width(c.as_str()));
        }
    }
    let (open, close) = matrix_brackets(&env);

    if display == Display::Inline || rows.len() == 1 {
        // Collapse: [a b; c d]
        let row_strs: Vec<String> = rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|c| c.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        return Some(MathRender::single(format!(
            "{open}{}{close}",
            row_strs.join("; ")
        )));
    }

    // Block: a padded grid with tall brackets.
    let body_lines: Vec<String> = rows
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .map(|(i, c)| pad_center(c, col_w[i]))
                .collect::<Vec<_>>()
                .join("  ")
        })
        .collect();
    Some(wrap_tall_brackets(body_lines, &env))
}

// ===========================================================================
// Super/subscript mapping
// ===========================================================================

/// Map a rendered string to Unicode superscripts. If EVERY grapheme is
/// mappable, returns the Unicode form; otherwise falls back to `^(…)` so nothing
/// renders as a wrong/blank glyph (P3 contract). A single mappable char is bare
/// (`x^2` → `x²`); a multi-char mappable run stays bare (`x^{10}` → `x¹⁰`).
fn to_superscript(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }
    match map_all(s, sup_char) {
        Some(mapped) => mapped,
        None => format!("^({s})"),
    }
}

/// Map a rendered string to Unicode subscripts, else `_(…)` (see [`to_superscript`]).
fn to_subscript(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }
    match map_all(s, sub_char) {
        Some(mapped) => mapped,
        None => format!("_({s})"),
    }
}

/// Map an index like `3` → `³` for `\sqrt[3]`. Falls back to `[n]` if unmappable.
fn sup_index(idx: &str) -> String {
    let t = idx.trim();
    match map_all(t, sup_char) {
        Some(m) => m,
        None => format!("[{t}]"),
    }
}

/// Apply `f` to every char of `s`; `Some(joined)` iff all chars mapped.
fn map_all(s: &str, f: fn(char) -> Option<char>) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        out.push(f(c)?);
    }
    Some(out)
}

/// A single char → its Unicode superscript, if one exists.
fn sup_char(c: char) -> Option<char> {
    Some(match c {
        '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
        '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹',
        '+' => '⁺', '-' => '⁻', '=' => '⁼', '(' => '⁽', ')' => '⁾',
        'n' => 'ⁿ', 'i' => 'ⁱ',
        'a' => 'ᵃ', 'b' => 'ᵇ', 'c' => 'ᶜ', 'd' => 'ᵈ', 'e' => 'ᵉ',
        'f' => 'ᶠ', 'g' => 'ᵍ', 'h' => 'ʰ', 'j' => 'ʲ', 'k' => 'ᵏ',
        'l' => 'ˡ', 'm' => 'ᵐ', 'o' => 'ᵒ', 'p' => 'ᵖ', 'r' => 'ʳ',
        's' => 'ˢ', 't' => 'ᵗ', 'u' => 'ᵘ', 'v' => 'ᵛ', 'w' => 'ʷ',
        'x' => 'ˣ', 'y' => 'ʸ', 'z' => 'ᶻ',
        ' ' => ' ',
        _ => return None,
    })
}

/// A single char → its Unicode subscript, if one exists.
fn sub_char(c: char) -> Option<char> {
    Some(match c {
        '0' => '₀', '1' => '₁', '2' => '₂', '3' => '₃', '4' => '₄',
        '5' => '₅', '6' => '₆', '7' => '₇', '8' => '₈', '9' => '₉',
        '+' => '₊', '-' => '₋', '=' => '₌', '(' => '₍', ')' => '₎',
        'a' => 'ₐ', 'e' => 'ₑ', 'h' => 'ₕ', 'i' => 'ᵢ', 'j' => 'ⱼ',
        'k' => 'ₖ', 'l' => 'ₗ', 'm' => 'ₘ', 'n' => 'ₙ', 'o' => 'ₒ',
        'p' => 'ₚ', 'r' => 'ᵣ', 's' => 'ₛ', 't' => 'ₜ', 'u' => 'ᵤ',
        'v' => 'ᵥ', 'x' => 'ₓ',
        ' ' => ' ',
        _ => return None,
    })
}

// ===========================================================================
// Symbol map (recon §6.3 — Greek, operators, relations, arrows, accents)
// ===========================================================================

/// Whether `\name` is a big operator that takes stacked limits.
fn is_bigop(name: &str) -> bool {
    matches!(
        name,
        "sum" | "prod" | "coprod" | "int" | "iint" | "iiint" | "oint"
            | "bigcup" | "bigcap" | "bigoplus" | "bigotimes" | "bigvee" | "bigwedge"
            | "lim"
    )
}

/// Whether `\name` is a combining accent.
fn is_accent(name: &str) -> bool {
    accent_mark(name).is_some()
}

/// The combining-mark codepoint for an accent command (applied AFTER the base).
fn accent_mark(name: &str) -> Option<&'static str> {
    Some(match name {
        "hat" | "widehat" => "\u{0302}",   // ̂
        "bar" | "overline" => "\u{0304}",  // ̄
        "vec" => "\u{20D7}",               // ⃗
        "dot" => "\u{0307}",               // ̇
        "ddot" => "\u{0308}",              // ̈
        "tilde" | "widetilde" => "\u{0303}", // ̃
        "check" => "\u{030C}",             // ̌
        "acute" => "\u{0301}",             // ́
        "grave" => "\u{0300}",             // ̀
        "breve" => "\u{0306}",             // ̆
        _ => return None,
    })
}

/// Map a bare `\command` (no backslash) to its Unicode glyph. Returns `None` for
/// commands the caller must keep literal. Covers Greek, relations, arithmetic /
/// set / logic operators, big-operator glyphs, calculus symbols, arrows, dots,
/// blackboard-bold, named functions (upright), and spacing commands.
fn symbol(name: &str) -> Option<String> {
    let g = match name {
        // ---- Greek (lower) ----
        "alpha" => "α", "beta" => "β", "gamma" => "γ", "delta" => "δ",
        "epsilon" => "ε", "varepsilon" => "ε", "zeta" => "ζ", "eta" => "η",
        "theta" => "θ", "vartheta" => "ϑ", "iota" => "ι", "kappa" => "κ",
        "lambda" => "λ", "mu" => "μ", "nu" => "ν", "xi" => "ξ",
        "omicron" => "ο", "pi" => "π", "varpi" => "ϖ", "rho" => "ρ",
        "varrho" => "ϱ", "sigma" => "σ", "varsigma" => "ς", "tau" => "τ",
        "upsilon" => "υ", "phi" => "φ", "varphi" => "φ", "chi" => "χ",
        "psi" => "ψ", "omega" => "ω",
        // ---- Greek (upper) ----
        "Gamma" => "Γ", "Delta" => "Δ", "Theta" => "Θ", "Lambda" => "Λ",
        "Xi" => "Ξ", "Pi" => "Π", "Sigma" => "Σ", "Upsilon" => "Υ",
        "Phi" => "Φ", "Psi" => "Ψ", "Omega" => "Ω",
        // ---- relations ----
        "leq" | "le" => "≤", "geq" | "ge" => "≥", "neq" | "ne" => "≠",
        "equiv" => "≡", "approx" => "≈", "cong" => "≅", "sim" => "∼",
        "simeq" => "≃", "propto" => "∝", "ll" => "≪", "gg" => "≫",
        "doteq" => "≐", "asymp" => "≍", "prec" => "≺", "succ" => "≻",
        "preceq" => "⪯", "succeq" => "⪰",
        // ---- arithmetic / set ----
        "pm" => "±", "mp" => "∓", "times" => "×", "div" => "÷", "cdot" => "·",
        "ast" => "∗", "star" => "⋆", "circ" => "∘", "bullet" => "•",
        "oplus" => "⊕", "ominus" => "⊖", "otimes" => "⊗", "oslash" => "⊘",
        "odot" => "⊙", "cap" => "∩", "cup" => "∪", "setminus" => "∖",
        "subset" => "⊂", "subseteq" => "⊆", "supset" => "⊃", "supseteq" => "⊇",
        "in" => "∈", "notin" => "∉", "ni" => "∋", "emptyset" => "∅",
        "varnothing" => "∅", "forall" => "∀", "exists" => "∃", "nexists" => "∄",
        // ---- logic ----
        "land" | "wedge" => "∧", "lor" | "vee" => "∨", "neg" | "lnot" => "¬",
        "top" => "⊤", "bot" => "⊥",
        // ---- big operators (bare glyph; limits handled structurally) ----
        "sum" => "∑", "prod" => "∏", "coprod" => "∐", "int" => "∫",
        "iint" => "∬", "iiint" => "∭", "oint" => "∮", "bigcup" => "⋃",
        "bigcap" => "⋂", "bigoplus" => "⨁", "bigotimes" => "⨂",
        "bigvee" => "⋁", "bigwedge" => "⋀",
        // ---- calculus / analysis ----
        "partial" => "∂", "nabla" => "∇", "infty" => "∞", "aleph" => "ℵ",
        "hbar" => "ℏ", "ell" => "ℓ", "Re" => "ℜ", "Im" => "ℑ", "wp" => "℘",
        "surd" => "√",
        // ---- dots / misc ----
        "ldots" | "dots" => "…", "cdots" => "⋯", "vdots" => "⋮", "ddots" => "⋱",
        "prime" => "′", "angle" => "∠", "perp" => "⊥", "parallel" => "∥",
        "mid" => "∣", "triangle" => "△", "square" => "□", "Box" => "□",
        "checkmark" => "✓",
        // ---- arrows ----
        "to" | "rightarrow" => "→", "gets" | "leftarrow" => "←",
        "leftrightarrow" => "↔", "Rightarrow" => "⇒", "Leftarrow" => "⇐",
        "Leftrightarrow" => "⇔", "uparrow" => "↑", "downarrow" => "↓",
        "updownarrow" => "↕", "mapsto" => "↦", "longrightarrow" => "⟶",
        "longleftarrow" => "⟵", "implies" => "⟹", "iff" => "⟺",
        "hookrightarrow" => "↪", "nearrow" => "↗", "searrow" => "↘",
        // ---- blackboard-bold (single-letter \mathbb shortcuts are rare; the
        //      common bare-cmd forms) ----
        "R" => "ℝ", "N" => "ℕ", "Z" => "ℤ", "Q" => "ℚ", "C" => "ℂ",
        // ---- named functions (render upright; the name as-is) ----
        "sin" | "cos" | "tan" | "cot" | "sec" | "csc" | "sinh" | "cosh"
        | "tanh" | "coth" | "arcsin" | "arccos" | "arctan" | "log" | "ln"
        | "lg" | "exp" | "det" | "dim" | "ker" | "deg" | "gcd" | "hom"
        | "arg" | "max" | "min" | "sup" | "inf" | "lim" | "Pr" | "mod" => name,
        // ---- spacing commands collapse to a space (or nothing) ----
        "quad" => " ", "qquad" => "  ", "," | ";" | ":" | " " => " ",
        "!" => "",
        // ---- escaped literals ($, {, }, %, &, #, _) keep the char ----
        "$" => "$", "{" => "{", "}" => "}", "%" => "%", "&" => "&",
        "#" => "#", "_" => "_",
        _ => return None,
    };
    Some(g.to_string())
}

// ===========================================================================
// Layout helpers (all display-cell aware)
// ===========================================================================

/// Whether `s` contains a TOP-LEVEL binary operator (not inside parentheses).
/// Used by `\frac` to decide whether a side needs parenthesizing in inline form
/// (`\frac{a+b}{c}` → `(a+b)/c`).
fn has_top_level_binop(s: &str) -> bool {
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '+' | '-' | '*' | '/' if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// Center `s` in `w` display cells (pad both sides; extra space goes right).
fn center(s: &str, w: usize) -> String {
    let sw = UnicodeWidthStr::width(s);
    if sw >= w {
        return s.to_string();
    }
    let total = w - sw;
    let left = total / 2;
    let right = total - left;
    format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
}

/// Center for matrix cells (alias of [`center`], named for call-site clarity).
fn pad_center(s: &str, w: usize) -> String {
    center(s, w)
}

/// Right-pad `s` to `w` display cells (no truncation if already wider).
fn pad_right(s: &str, w: usize) -> String {
    let sw = UnicodeWidthStr::width(s);
    if sw >= w {
        return s.to_string();
    }
    format!("{}{}", s, " ".repeat(w - sw))
}

/// The flattened text of a group of `Char` tokens (for env names like `pmatrix`).
fn group_text(tokens: &[Tok]) -> String {
    tokens
        .iter()
        .map(|t| match t {
            Tok::Char(c) => c.to_string(),
            _ => String::new(),
        })
        .collect()
}

/// Whether `env` is a supported matrix environment.
fn is_matrix_env(env: &str) -> bool {
    matches!(
        env,
        "matrix" | "pmatrix" | "bmatrix" | "Bmatrix" | "vmatrix" | "Vmatrix"
    )
}

/// The (open, close) inline bracket strings for a matrix environment.
fn matrix_brackets(env: &str) -> (&'static str, &'static str) {
    match env {
        "pmatrix" => ("(", ")"),
        "bmatrix" => ("[", "]"),
        "Bmatrix" => ("{", "}"),
        "vmatrix" => ("|", "|"),
        "Vmatrix" => ("‖", "‖"),
        _ => ("", ""), // plain matrix: no brackets
    }
}

/// Wrap a block matrix body in tall extensible brackets (box-drawing pieces).
fn wrap_tall_brackets(body: Vec<String>, env: &str) -> MathRender {
    let n = body.len();
    let inner_w = body
        .iter()
        .map(|l| UnicodeWidthStr::width(l.as_str()))
        .max()
        .unwrap_or(0);
    let (left, right): (Vec<&str>, Vec<&str>) = bracket_pieces(env, n);
    let mut out: Vec<String> = Vec::with_capacity(n);
    for (i, line) in body.iter().enumerate() {
        let padded = pad_right(line, inner_w);
        out.push(format!("{}{}{}", left[i], padded, right[i]));
    }
    let baseline = n / 2;
    MathRender::multi(out, baseline)
}

/// Per-row left/right bracket pieces for an `n`-row tall bracket. Uses the
/// extensible box-drawing bracket parts so a 2-row bracket is `⎡…⎤` / `⎣…⎦`.
fn bracket_pieces(env: &str, n: usize) -> (Vec<&'static str>, Vec<&'static str>) {
    // Return per-row pieces for the given env. For n==1 just the single-line
    // bracket; for n>=2 the top/middle/bottom extensible pieces.
    let (lt, lm, lb, ls, rt, rm, rb, rs) = match env {
        "pmatrix" => ("⎛", "⎜", "⎝", "(", "⎞", "⎟", "⎠", ")"),
        "bmatrix" => ("⎡", "⎢", "⎣", "[", "⎤", "⎥", "⎦", "]"),
        "Bmatrix" => ("⎧", "⎨", "⎩", "{", "⎫", "⎬", "⎭", "}"),
        "vmatrix" => ("│", "│", "│", "|", "│", "│", "│", "|"),
        "Vmatrix" => ("‖", "‖", "‖", "‖", "‖", "‖", "‖", "‖"),
        _ => (" ", " ", " ", " ", " ", " ", " ", " "),
    };
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![ls], vec![rs]);
    }
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        if i == 0 {
            left.push(lt);
            right.push(rt);
        } else if i == n - 1 {
            left.push(lb);
            right.push(rb);
        } else {
            left.push(lm);
            right.push(rm);
        }
    }
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inline(s: &str) -> String {
        latex_to_unicode(s, Display::Inline).lines.join("\n")
    }
    fn block(s: &str) -> MathRender {
        latex_to_unicode(s, Display::Block)
    }

    // ---- math_greek: \alpha -> α (the named test) -------------------------
    #[test]
    fn math_greek() {
        assert_eq!(inline("\\alpha"), "α");
        assert_eq!(inline("\\beta"), "β");
        assert_eq!(inline("\\gamma"), "γ");
        assert_eq!(inline("\\Omega"), "Ω");
        // In an expression with spacing preserved.
        assert_eq!(inline("\\alpha + \\beta"), "α + β");
        // A space after a command name terminates it and is kept literal (LaTeX
        // semantics): `\Delta x` → `Δ x`, while a braced arg has no gap.
        assert_eq!(inline("\\Delta x"), "Δ x");
        assert_eq!(inline("\\Delta{x}"), "Δx");
        // Unknown command is kept literal (never silently dropped).
        assert_eq!(inline("\\notacommand"), "\\notacommand");
    }

    // ---- math_operators (the named test) ----------------------------------
    #[test]
    fn math_operators() {
        assert_eq!(inline("a \\leq b"), "a ≤ b");
        assert_eq!(inline("a \\geq b"), "a ≥ b");
        assert_eq!(inline("a \\neq b"), "a ≠ b");
        assert_eq!(inline("\\pm"), "±");
        assert_eq!(inline("a \\times b"), "a × b");
        assert_eq!(inline("a \\div b"), "a ÷ b");
        assert_eq!(inline("\\sum"), "∑");
        assert_eq!(inline("\\prod"), "∏");
        assert_eq!(inline("\\int"), "∫");
        assert_eq!(inline("\\sqrt{2}"), "√(2)");
        assert_eq!(inline("\\infty"), "∞");
        assert_eq!(inline("\\partial"), "∂");
        assert_eq!(inline("\\nabla"), "∇");
        assert_eq!(inline("x \\to y"), "x → y");
        assert_eq!(inline("p \\Rightarrow q"), "p ⇒ q");
        // The recon §6.9 multi-operator fixture.
        assert_eq!(inline("a \\leq b \\neq c"), "a ≤ b ≠ c");
    }

    // ---- math_frac_block: 3-line stack (the named test) -------------------
    #[test]
    fn math_frac_block() {
        let r = block("\\frac{a}{b}");
        assert_eq!(r.lines.len(), 3, "block frac is a 3-line stack");
        assert_eq!(r.lines[0].trim(), "a");
        // The middle row is the fraction bar (U+2500), at least one cell wide.
        assert!(r.lines[1].chars().all(|c| c == '─'));
        assert!(!r.lines[1].is_empty());
        assert_eq!(r.lines[2].trim(), "b");
        assert_eq!(r.baseline, 1, "baseline is the bar row");

        // A wider numerator parenthesizes nothing in block form but widens the bar.
        let r2 = block("\\frac{a+b}{c}");
        assert_eq!(r2.lines.len(), 3);
        assert_eq!(r2.lines[0].trim(), "a+b");
        assert_eq!(UnicodeWidthStr::width(r2.lines[1].as_str()), 3); // bar spans "a+b"
        assert_eq!(r2.lines[2].trim(), "c");

        // Inline frac collapses to a/b and parenthesizes a binop side.
        assert_eq!(inline("\\frac{a}{b}"), "a/b");
        assert_eq!(inline("\\frac{a+b}{c}"), "(a+b)/c");
        assert_eq!(inline("\\frac{1}{2}"), "1/2");
    }

    // ---- super/subscripts -------------------------------------------------
    #[test]
    fn math_super_subscript() {
        assert_eq!(inline("x^2"), "x²");
        assert_eq!(inline("y_1"), "y₁");
        assert_eq!(inline("x^2 + y_1"), "x² + y₁");
        assert_eq!(inline("x^{10}"), "x¹⁰");
        assert_eq!(inline("a_{i}"), "aᵢ");
        // `m`,`a`,`x` all HAVE Unicode subscripts, so x_{max} maps fully.
        assert_eq!(inline("x_{max}"), "xₘₐₓ");
        // Unmappable superscript (q has no Unicode form) → ^(…) fallback.
        assert_eq!(inline("x^q"), "x^(q)");
        // A subscript with a consonant lacking a Unicode form (b) → _(…) fallback,
        // since map_all requires EVERY char to map.
        assert_eq!(inline("x_{ab}"), "x_(ab)");
    }

    // ---- sqrt -------------------------------------------------------------
    #[test]
    fn math_sqrt() {
        assert_eq!(inline("\\sqrt{x}"), "√(x)");
        assert_eq!(inline("\\sqrt[3]{x}"), "³√(x)");
        // Block sqrt: 2-line overbar + "√ x".
        let r = block("\\sqrt{x}");
        assert_eq!(r.lines.len(), 2);
        assert!(r.lines[1].contains('√'));
        assert!(r.lines[1].contains('x'));
    }

    // ---- big operators with limits ----------------------------------------
    #[test]
    fn math_bigop_limits() {
        // Inline: limits as sub/superscripts beside the glyph.
        assert_eq!(inline("\\sum_{i=0}^{n} i"), "∑ᵢ₌₀ⁿ i");
        assert_eq!(inline("\\int_0^1 f"), "∫₀¹ f");
        // Block: stacked hi / glyph / lo.
        let r = block("\\sum_{i=0}^{n}");
        assert_eq!(r.lines.len(), 3);
        assert_eq!(r.lines[0].trim(), "n");
        assert_eq!(r.lines[1].trim(), "∑");
        assert_eq!(r.lines[2].trim(), "i=0");
    }

    // ---- accents ----------------------------------------------------------
    #[test]
    fn math_accents() {
        // Combining marks: base char + combining codepoint.
        assert_eq!(inline("\\hat{x}"), "x\u{0302}");
        assert_eq!(inline("\\bar{x}"), "x\u{0304}");
        assert_eq!(inline("\\vec{v}"), "v\u{20D7}");
        // Overline-style accent spans every grapheme of a multi-char base.
        assert_eq!(inline("\\bar{AB}"), "A\u{0304}B\u{0304}");
    }

    // ---- matrices ---------------------------------------------------------
    #[test]
    fn math_matrix() {
        // Inline pmatrix collapses to (a b; c d).
        assert_eq!(
            inline("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}"),
            "(1 2; 3 4)"
        );
        assert_eq!(
            inline("\\begin{bmatrix}a&b\\\\c&d\\end{bmatrix}"),
            "[a b; c d]"
        );
        // Block pmatrix renders multiple rows wrapped in tall brackets.
        let r = block("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}");
        assert_eq!(r.lines.len(), 2);
        assert!(r.lines[0].starts_with('⎛'));
        assert!(r.lines[1].starts_with('⎝'));
    }

    // ---- the never-panic / never-corrupt contract -------------------------
    #[test]
    fn math_degrades_never_panics() {
        // Unbalanced braces don't panic; output is best-effort, never lost.
        let _ = inline("\\frac{a}{");
        let _ = inline("{{{{");
        let _ = inline("}}}}");
        let _ = inline("\\sqrt[");
        let _ = inline("\\begin{pmatrix}1&2");
        let _ = inline("^^^___");
        let _ = inline("\\");
        // CJK and emoji inside math don't break width math.
        let _ = inline("\\text{你好} + x");
        // Empty / whitespace is one empty line.
        assert_eq!(inline(""), "");
        assert_eq!(inline("   "), "");
    }

    #[test]
    fn math_inline_is_single_line() {
        // Inline NEVER returns more than one line, even for frac/sqrt/matrix.
        for src in [
            "\\frac{a}{b}",
            "\\sqrt{x}",
            "\\sum_{i=0}^{n}",
            "\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}",
        ] {
            let r = latex_to_unicode(src, Display::Inline);
            assert_eq!(r.lines.len(), 1, "inline {src} must be one line");
        }
    }

    #[test]
    fn math_width_is_display_cells() {
        // The reported width is display cells, not bytes/chars.
        let r = latex_to_unicode("\\alpha\\beta", Display::Inline);
        assert_eq!(r.lines[0], "αβ");
        // Each Greek letter is 1 display cell.
        assert_eq!(r.width, 2);
    }
}
