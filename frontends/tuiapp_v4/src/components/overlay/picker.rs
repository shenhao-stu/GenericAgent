//! overlay/picker.rs — the reusable list picker card (`/llm` etc) + the unified
//! ask_user card (single / multi / numeric). All load-bearing list/selection
//! logic lives in [`crate::components::picker`] (pure + tested); this only PAINTS.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use ratatui::Frame;

use crate::components::clip_to;
use crate::components::picker::{AskMode, AskUserPicker, Picker};
use crate::i18n::{self, Lang};
use crate::theme::{Theme, Token};

use super::titled_block;

/// The reusable list picker card (`/llm` etc). Rows show `●` for the current item,
/// `[x]`/`[ ]` checkboxes in multi-select mode, and a `❯` cursor on the selection.
pub(crate) fn render_picker(frame: &mut Frame, area: Rect, picker: &Picker, theme: &Theme, lang: Lang) {
    let multi = picker.kind.multi();
    let max_label = picker
        .items
        .iter()
        .map(|i| {
            unicode_width::UnicodeWidthStr::width(i.label.as_str())
                + unicode_width::UnicodeWidthStr::width(i.detail.as_str())
                + 4
        })
        .max()
        .unwrap_or(20);
    let title = picker.kind.title(lang);
    let title_w = unicode_width::UnicodeWidthStr::width(title) + 4;
    let inner_w = max_label.max(title_w).max(28) as u16;
    let (start, slice) = picker.window();
    let w = (inner_w + 6).min(area.width.saturating_sub(2)).max(20);
    let h = (slice.len() as u16 + 4).min(area.height.saturating_sub(2)).max(5);
    let card = super::centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = titled_block(title, theme);
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let mut lines: Vec<Line> = Vec::with_capacity(slice.len() + 1);
    if picker.is_empty() {
        lines.push(Line::from(Span::styled(
            i18n::t(lang, "picker.empty"),
            Style::default().fg(theme.color(Token::Dim)),
        )));
    }
    for (i, item) in slice.iter().enumerate() {
        let idx = start + i;
        let selected = idx == picker.sel;
        let mut spans: Vec<Span> = Vec::new();
        spans.push(Span::styled(
            if selected { "❯ " } else { "  " },
            Style::default().fg(theme.color(Token::Suggestion)),
        ));
        if multi {
            let box_ = if item.checked { "[x] " } else { "[ ] " };
            let tok = if item.checked { Token::Success } else { Token::Dim };
            spans.push(Span::styled(box_, Style::default().fg(theme.color(tok))));
        } else if item.current {
            spans.push(Span::styled("● ", Style::default().fg(theme.color(Token::Success))));
        } else {
            spans.push(Span::raw("  "));
        }
        let label_tok = if selected { Token::Suggestion } else { Token::Text };
        let label_mod = if selected || item.current {
            Modifier::BOLD
        } else {
            Modifier::empty()
        };
        spans.push(Span::styled(
            item.label.clone(),
            Style::default().fg(theme.color(label_tok)).add_modifier(label_mod),
        ));
        if !item.detail.is_empty() {
            spans.push(Span::styled(
                format!("   {}", item.detail),
                Style::default().fg(theme.color(Token::Dim)),
            ));
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(Span::styled(
        format!("  {}", picker.kind.hint(lang)),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The unified ask_user card (§7): question + candidate rows + an inline free-text
/// row, with multi-select checkboxes / numeric ordinals as the mode dictates.
pub(crate) fn render_ask_user(frame: &mut Frame, area: Rect, ask: &AskUserPicker, theme: &Theme, lang: Lang) {
    let title = match ask.mode {
        AskMode::Single => i18n::t(lang, "ask.title.single"),
        AskMode::Multi => i18n::t(lang, "ask.title.multi"),
        AskMode::Numeric => i18n::t(lang, "ask.title.numeric"),
    };
    let w = (area.width.saturating_sub(8)).clamp(30, 96);
    // Height: question (wrapped, ~3) + candidates + free-text + hint + chrome.
    let body_rows = ask.candidates.len() as u16 + if ask.free_text { 1 } else { 0 } + 5;
    let h = body_rows.min(area.height.saturating_sub(2)).max(7);
    let card = super::centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = titled_block(title, theme);
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let inner_w = inner.width as usize;
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        clip_to(&ask.question, inner_w.saturating_sub(2)),
        Style::default().fg(theme.color(Token::Text)).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    for (i, cand) in ask.candidates.iter().enumerate() {
        let selected = ask.sel == i;
        let mut spans: Vec<Span> = Vec::new();
        spans.push(Span::styled(
            if selected { "❯ " } else { "  " },
            Style::default().fg(theme.color(Token::Suggestion)),
        ));
        match ask.mode {
            AskMode::Multi => {
                let on = ask.checked.get(i).copied().unwrap_or(false);
                let box_ = if on { "[x] " } else { "[ ] " };
                let tok = if on { Token::Success } else { Token::Dim };
                spans.push(Span::styled(box_, Style::default().fg(theme.color(tok))));
            }
            AskMode::Numeric => {
                spans.push(Span::styled(
                    format!("{}. ", i + 1),
                    Style::default().fg(theme.color(Token::Suggestion)),
                ));
            }
            AskMode::Single => {}
        }
        let tok = if selected { Token::Suggestion } else { Token::Text };
        spans.push(Span::styled(
            clip_to(cand, inner_w.saturating_sub(6)),
            Style::default().fg(theme.color(tok)).add_modifier(if selected {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
        ));
        lines.push(Line::from(spans));
    }

    if ask.free_text || ask.mode == AskMode::Numeric {
        let on_input = ask.on_free_text_row() || ask.mode == AskMode::Numeric;
        let label = if ask.mode == AskMode::Numeric {
            i18n::t(lang, "ask.input.number")
        } else {
            "› "
        };
        let mut spans: Vec<Span> = vec![
            Span::styled(
                if on_input { "❯ " } else { "  " },
                Style::default().fg(theme.color(Token::Suggestion)),
            ),
            Span::styled(label, Style::default().fg(theme.color(Token::Dim))),
        ];
        if ask.input.is_empty() {
            spans.push(Span::styled(
                i18n::t(lang, "ask.input.placeholder"),
                Style::default().fg(theme.color(Token::Dim)),
            ));
        } else {
            spans.push(Span::styled(
                ask.input.clone(),
                Style::default().fg(theme.color(Token::Text)),
            ));
        }
        if on_input {
            spans.push(Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)));
        }
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));
    let hint = match ask.mode {
        AskMode::Multi => i18n::t(lang, "ask.hint.multi"),
        AskMode::Numeric => i18n::t(lang, "ask.hint.numeric"),
        AskMode::Single => i18n::t(lang, "ask.hint.single"),
    };
    lines.push(Line::from(Span::styled(
        hint,
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}
