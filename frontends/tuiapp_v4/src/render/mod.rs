//! render/ — the render plane (checklist §3): the logical [`block::Block`] model
//! (SOURCE text, the only width-independent truth), the CJK-aware soft-wrap +
//! per-`(block_id,width)` cache with prefix sums ([`measure`]), the logical
//! [`viewport::ScrollAnchor`] that makes resize drift-free (P1), and clean
//! logical-source clipboard copy via OSC 52 + native + copy-mode ([`copy`], P2).
//!
//! The two pain points are solved *by construction* here:
//!   * **P1** — scroll is a logical `(block_id, intra)` anchor (or `Bottom`);
//!     visual rows are a pure, memoized function of `(block, width)`; a resize
//!     re-derives the window from the same anchor → zero drift.
//!   * **P2** — copy reads `block.source`, never rendered rows, and even the
//!     row-reconstruction helper joins soft-wrap continuations with no `\n`.
//!
//! All load-bearing logic is pure and unit-tested in the submodules; ratatui
//! widgets just project this state.

pub mod block;
pub mod chip;
pub mod copy;
pub mod fold;
pub mod measure;
pub mod viewport;

// The render plane's public API surface. Some items are consumed by later phases
// (the session dashboard, the copy-mode overlay) rather than the Phase-2 wiring,
// so they are re-exported here for discoverability even when not yet referenced
// crate-wide.
#[allow(unused_imports)]
pub use block::{Block, BlockId, BlockRole};
#[allow(unused_imports)]
pub use copy::{CopyCaps, CopyMethod, CopyResult, Selection};
#[allow(unused_imports)]
pub use measure::{VisualLine, WrapCache};
#[allow(unused_imports)]
pub use viewport::{ScrollAnchor, Viewport};
