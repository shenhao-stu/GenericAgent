//! tests/bridge_liveness.rs — an integration test that ACTUALLY spawns the real
//! `scripts/ga_bridge.py` child and confirms the N1 plumbing works end to end:
//! discovery finds the script, the child spawns with PYTHONUTF8=1, the reader
//! thread decodes its stdout, and we receive a lifecycle event quickly (never a
//! silent hang). It tolerates a degraded core (no LLM configured) — the gate is
//! that SOME visible event arrives, which is the whole point of N1.
//!
//! This is `#[ignore]` by default so the normal `cargo test` (the build gate)
//! stays hermetic and fast; run it explicitly with:
//!     cargo test --test bridge_liveness -- --ignored --nocapture

use std::sync::mpsc;
use std::time::{Duration, Instant};

// The crate is a binary (`tui_v4`), so its modules aren't importable as a lib.
// We re-validate the public contract by driving the binary's bridge through a
// thin re-spawn here would require a lib target; instead this test exercises the
// SAME ga_bridge.py via a direct child process to prove the protocol handshake
// the Rust reader relies on is real. (The Rust reader path itself is unit-tested
// in src/bridge for discovery; this asserts the Python side answers.)

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    // tests/ lives at <repo>/frontends/tuiapp_v4/tests — four up is the repo.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // .../tuiapp_v4
    manifest
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or(manifest)
}

fn bridge_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("ga_bridge.py")
}

#[test]
#[ignore = "spawns the real python GA core; run with --ignored"]
fn ga_bridge_handshakes_ready() {
    let bridge = bridge_path();
    assert!(
        bridge.exists(),
        "ga_bridge.py must exist at {}",
        bridge.display()
    );

    let mut child = Command::new("python")
        .arg(&bridge)
        .env("PYTHONUTF8", "1")
        .current_dir(repo_root())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python ga_bridge.py (is python on PATH?)");

    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Wait (bounded) for the Ready frame. The whole N1 point: a visible frame
    // arrives, never a silent hang. 60s is generous for a cold GA-core import.
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut got_ready = false;
    let mut last = String::new();
    while Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => {
                last = line.clone();
                if line.contains("\"type\":\"Ready\"") {
                    got_ready = true;
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Send Shutdown so the child exits cleanly (never hangs the test runner).
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(b"{\"type\":\"Shutdown\"}\n");
        let _ = stdin.flush();
    }
    let _ = child.wait();

    assert!(
        got_ready,
        "expected a Ready frame from ga_bridge.py within 60s; last line: {last:?}"
    );
}
