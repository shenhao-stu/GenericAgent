//! app/overlay_ops.rs — the overlay-stack constructors (§3 / §7): opening +
//! folding the modal pickers, the ask_user card + its FIFO queue, the `/btw`
//! answer routing, and the simple info overlays. All pure-ish `impl AppState`
//! methods (in-memory only); the renderer in `components::overlay` PAINTS them.

use crate::app::{AppState, Overlay};
use crate::bridge::protocol::LlmItem;
use crate::components::picker::{AskUserPicker, PickItem, Picker, PickerKind};

impl AppState {
    /// Fold an incoming `LlmList` frame (N3): if a `/llm` picker is already open
    /// (the `querying…` placeholder), REPLACE its items in place; otherwise open
    /// a fresh `/llm` picker. The selection seeds on the current model.
    pub fn apply_llm_list(&mut self, items: Vec<LlmItem>) {
        let pick_items = llm_items_to_picks(&items);
        match &mut self.overlay {
            Some(Overlay::Picker { picker, .. }) if picker.kind == PickerKind::Llm => {
                let sel = pick_items.iter().position(|i| i.current).unwrap_or(0);
                picker.items = pick_items;
                picker.sel = sel;
            }
            _ => {
                self.overlay = Some(Overlay::Picker {
                    picker: Picker::new(PickerKind::Llm, pick_items),
                    theme_backup: None,
                });
            }
        }
    }

    /// Open a reusable list picker overlay (`/theme` `/emoji` `/language`
    /// `/export` `/rewind` `/continue` `/scheduler`). `theme_backup` is the theme
    /// to restore on `Esc` for a live-preview picker (else `None`).
    pub fn open_picker(&mut self, picker: Picker, theme_backup: Option<Theme>) {
        self.overlay = Some(Overlay::Picker { picker, theme_backup });
    }

    /// Fold an incoming `BtwAnswer` frame into the `/btw` card (§7): set the
    /// answer IFF the card is still open AND its `ask_id` matches (a stale card
    /// can't show a newer side-question's answer). The answer is shown in the
    /// EPHEMERAL card only — NEVER pushed to the transcript. No matching card
    /// (dismissed with Esc) → silently dropped (the "no history pollution"
    /// contract).
    pub fn apply_btw_answer(&mut self, ask_id: String, text: Option<String>, error: Option<String>) {
        if let Some(Overlay::Btw { ask_id: open_id, answer, .. }) = self.overlay.as_mut() {
            if *open_id == ask_id {
                let body = match (text, error) {
                    (Some(t), _) => t,
                    (None, Some(e)) => format!("{}: {e}", crate::i18n::t(self.lang, "btw.failed")),
                    (None, None) => crate::i18n::tf(self.lang, "btw.failed"),
                };
                *answer = Some(body);
            }
        }
    }

    /// Open the unified ask_user card from a pending ask (§7).
    pub fn open_ask_user(&mut self) {
        if let Some(ask) = self.pending_ask.clone() {
            let candidates: Vec<String> = ask.options.iter().map(|o| o.label.clone()).collect();
            self.overlay = Some(Overlay::AskUser(AskUserPicker::new(
                ask.ask_id,
                ask.question,
                candidates,
                ask.free_text,
            )));
        }
    }

    /// After the current ask is answered/dismissed, surface the NEXT queued ask
    /// (if any) — opening its card. The caller clears `pending_ask` first; this
    /// pops the queue into it. Returns `true` if one was surfaced. This is what
    /// makes "queued parallel asks surface in turn" (§7) work.
    pub fn surface_next_ask(&mut self) -> bool {
        if self.pending_ask.is_some() {
            return false; // one is still active.
        }
        if let Some(next) = self.ask_queue.pop_front() {
            self.pending_ask = Some(next);
            if self.overlay.is_none() {
                self.open_ask_user();
            }
            true
        } else {
            false
        }
    }

    /// Open a simple info overlay (help / status / cost / verbose / btw).
    pub fn open_overlay(&mut self, overlay: Overlay) {
        self.overlay = Some(overlay);
    }

    /// Close any open overlay (the universal `Esc` for modals). Returns `true` if
    /// one was open (so the key handler knows the Esc was consumed by a modal).
    pub fn close_overlay(&mut self) -> bool {
        self.overlay.take().is_some()
    }

    /// True while ANY overlay is up. (The key router branches on
    /// [`Overlay::is_modal`] so a non-modal `/btw` toast doesn't steal input;
    /// this stays the simple "is something open" query for callers/tests.)
    #[allow(dead_code)]
    pub fn has_overlay(&self) -> bool {
        self.overlay.is_some()
    }
}

use crate::theme::Theme;

/// Map `LlmList` items `(idx, name, current)` onto picker rows: the row label is
/// `"i. name"` (the widget prepends `●` for the current one — §4), the `id` is
/// the 0-based LLM index (`SwitchLlm` is `id+1`), and `current` carries the
/// active marker. PURE — the `/llm` picker mapping pinned by `llm_picker_maps_index`.
pub fn llm_items_to_picks(items: &[LlmItem]) -> Vec<PickItem> {
    items
        .iter()
        .map(|it| {
            PickItem::new(it.idx as usize, format!("{}. {}", it.idx, it.name)).current(it.current)
        })
        .collect()
}
