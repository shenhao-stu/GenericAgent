//! components/ — the ratatui view layer.
//!
//! The cockpit layout (header / transcript / composer / footer / dropdown) lives
//! in the [`cockpit`] submodule; its pure label/format helpers live in [`text`].
//! The full-screen views (dashboard / workflow panel), modal overlays, the
//! pickers, and the effects painter are sibling submodules. The single
//! immediate-mode entry point is [`render`] (re-exported from [`cockpit`]).

pub mod cockpit;
pub mod continue_picker;
pub mod dashboard;
pub mod effects_paint;
pub mod overlay;
pub mod picker;
pub mod scheduler;
pub mod text;

pub use cockpit::render;

// The pure label helpers moved into `text`; re-export the ones the sibling view
// submodules reach via `super::` (dashboard / overlay / continue_picker) so their
// call sites keep resolving unchanged.
pub(crate) use text::{clip_to, compact_cwd, truncate_model, MODEL_LABEL_CAP};
