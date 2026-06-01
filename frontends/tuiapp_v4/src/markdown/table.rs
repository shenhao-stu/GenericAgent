//! markdown/table.rs — GFM table buffering + width-aligned emission. The
//! [`Walker`](super::render) collects cells into a [`TableBuf`]; [`emit_table`]
//! turns it into per-column display-width-aligned rows honoring the
//! `:---`/`:--:`/`---:` alignment spec. Alignment uses display width (never
//! `.len()`) so CJK/emoji columns line up.

use pulldown_cmark::Alignment;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::theme::{Theme, Token};

pub(crate) struct TableBuf {
    pub(crate) alignments: Vec<Alignment>,
    /// Header cells, then body rows; each cell is its plain display string.
    pub(crate) header: Vec<String>,
    pub(crate) rows: Vec<Vec<String>>,
    /// Accumulator for the cell currently being built.
    pub(crate) cur_cell: String,
    /// Accumulator for the row currently being built (body).
    pub(crate) cur_row: Vec<String>,
    /// True while reading the header row.
    pub(crate) in_header: bool,
}

fn col(theme: &Theme, tok: Token) -> Style {
    Style::default().fg(theme.color(tok))
}

/// Emit a GFM table with per-column display-width alignment. Header row, a `─┼─`
/// rule, body rows. Empty (no columns) → no lines.
pub(crate) fn emit_table(theme: &Theme, tb: &TableBuf) -> Vec<Line<'static>> {
    let ncols = tb
        .header
        .len()
        .max(tb.rows.iter().map(|r| r.len()).max().unwrap_or(0));
    if ncols == 0 {
        return Vec::new();
    }
    // Per-column display width = max over header + all body cells.
    let mut widths = vec![0usize; ncols];
    let consider = |row: &[String], widths: &mut Vec<usize>| {
        for (i, c) in row.iter().enumerate() {
            if i < ncols {
                widths[i] = widths[i].max(UnicodeWidthStr::width(c.as_str()));
            }
        }
    };
    consider(&tb.header, &mut widths);
    for r in &tb.rows {
        consider(r, &mut widths);
    }

    let align = |i: usize| -> Alignment { tb.alignments.get(i).copied().unwrap_or(Alignment::None) };

    let mut out: Vec<Line<'static>> = Vec::new();

    // Header (bold).
    let header_style = col(theme, Token::Text).add_modifier(Modifier::BOLD);
    out.push(table_row(theme, &tb.header, &widths, ncols, &align, header_style));

    // Separator rule: `─` per column joined by `┼`.
    let mut sep_spans: Vec<Span<'static>> = Vec::new();
    sep_spans.push(Span::styled("├─".to_string(), col(theme, Token::Border)));
    for i in 0..ncols {
        if i > 0 {
            sep_spans.push(Span::styled("─┼─".to_string(), col(theme, Token::Border)));
        }
        sep_spans.push(Span::styled("─".repeat(widths[i]), col(theme, Token::Border)));
    }
    sep_spans.push(Span::styled("─┤".to_string(), col(theme, Token::Border)));
    out.push(Line::from(sep_spans));

    // Body rows.
    let body_style = col(theme, Token::Text);
    for r in &tb.rows {
        out.push(table_row(theme, r, &widths, ncols, &align, body_style));
    }
    out
}

/// Build one rendered table row: `│ ` borders, each cell aligned in its column's
/// display width per the column alignment.
fn table_row(
    theme: &Theme,
    cells: &[String],
    widths: &[usize],
    ncols: usize,
    align: &dyn Fn(usize) -> Alignment,
    cell_style: Style,
) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    spans.push(Span::styled("│ ".to_string(), col(theme, Token::Border)));
    for i in 0..ncols {
        if i > 0 {
            spans.push(Span::styled(" │ ".to_string(), col(theme, Token::Border)));
        }
        let empty = String::new();
        let cell = cells.get(i).unwrap_or(&empty);
        let padded = pad_cell(cell, widths[i], align(i));
        spans.push(Span::styled(padded, cell_style));
    }
    spans.push(Span::styled(" │".to_string(), col(theme, Token::Border)));
    Line::from(spans)
}

/// Pad `cell` to `width` display cells per `align` (default = left). Uses display
/// width, never `.len()`, so CJK/emoji columns line up.
fn pad_cell(cell: &str, width: usize, align: Alignment) -> String {
    let cw = UnicodeWidthStr::width(cell);
    if cw >= width {
        return cell.to_string();
    }
    let pad = width - cw;
    match align {
        Alignment::Right => format!("{}{}", " ".repeat(pad), cell),
        Alignment::Center => {
            let l = pad / 2;
            let r = pad - l;
            format!("{}{}{}", " ".repeat(l), cell, " ".repeat(r))
        }
        _ => format!("{}{}", cell, " ".repeat(pad)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::render::render_markdown;
    use crate::theme::Theme;
    use unicode_width::UnicodeWidthStr;

    fn plain(theme: &Theme, src: &str) -> Vec<String> {
        render_markdown(src, theme)
            .into_iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    #[test]
    fn md_table() {
        let theme = Theme::default_theme();
        let src = "\
| Name | Score |
|:-----|------:|
| Alice | 90 |
| Bob | 100 |
";
        let lines = plain(&theme, src);
        // Header + separator + 2 body rows = 4 content rows (no extra noise).
        assert_eq!(lines.len(), 4, "table = header + rule + 2 rows; got {lines:?}");

        // Header carries both column titles.
        assert!(lines[0].contains("Name"));
        assert!(lines[0].contains("Score"));
        // The rule row is box-drawing.
        assert!(lines[1].contains('┼') || lines[1].contains('─'));

        // Right-aligned "Score" column: "90" is padded on the LEFT to width 5
        // (max of "Score"=5, "100"=3), so the body cell shows "  90".
        let alice = &lines[2];
        assert!(alice.contains("Alice"));
        assert!(alice.contains("  90"), "right-aligned score: {alice:?}");
        let bob = &lines[3];
        assert!(bob.contains("100"));

        // Every rendered row has the SAME display width (columns line up).
        let w0 = UnicodeWidthStr::width(lines[0].as_str());
        for l in &lines {
            assert_eq!(
                UnicodeWidthStr::width(l.as_str()),
                w0,
                "all table rows must share a width: {l:?}"
            );
        }
    }

    #[test]
    fn md_table_cjk_alignment() {
        let theme = Theme::default_theme();
        // CJK cells are 2 cells/char — alignment must use display width.
        let src = "\
| 名字 | 分数 |
|------|------|
| 张三 | 90 |
";
        let lines = plain(&theme, src);
        let w0 = UnicodeWidthStr::width(lines[0].as_str());
        for l in &lines {
            assert_eq!(UnicodeWidthStr::width(l.as_str()), w0);
        }
    }
}
