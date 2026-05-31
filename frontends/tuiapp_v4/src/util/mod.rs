//! util/ — small cross-cutting helpers (OSC escape sequences, terminal caps).
//! Kept pure where it matters so the load-bearing payload bytes are unit-tested
//! (the actual write to fd 1 is a thin effectful wrapper).

pub mod osc;
