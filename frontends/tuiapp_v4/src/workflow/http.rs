//! workflow/http.rs — a MINIMAL, dependency-free HTTP/1.0 client for the
//! `127.0.0.1`-only conductor + hive polling (checklist §7; recon §2.3 / §3.2).
//!
//! Why hand-rolled: the panel only ever talks to LOCAL services (`conductor.py`
//! on `:8900`, an `agent_bbs.py` board on `:<port>`) over plain HTTP — no TLS, no
//! redirects, no chunked uploads. Pulling in `reqwest`/`ureq` (and their tokio /
//! TLS / async transitive deps) for two localhost JSON calls is unjustified
//! weight. A ~120-line `TcpStream` GET/POST with hard connect + read TIMEOUTS is
//! enough AND keeps the §3 contract that the watcher NEVER blocks the chat: every
//! socket op has a bounded timeout, so a dead/hung server fails fast into a
//! "down" result instead of stalling a thread.
//!
//! Scope guard: this is for trusted localhost only. It does not validate TLS,
//! follow redirects, or stream large bodies — it caps the response body. The
//! request/response PARSING is split into pure helpers so it is unit-tested
//! without a socket.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// The connect + read/write timeout for every localhost call. Short on purpose:
/// a hung conductor/BBS must NOT stall the watcher thread (let alone the chat) —
/// it fails into a `None`/down result well within one poll tick.
pub const HTTP_TIMEOUT: Duration = Duration::from_millis(1200);

/// Max response body bytes we keep (a board feed / subagent list is small; bound
/// a pathological server). 1 MiB is plenty for a localhost JSON poll.
pub const MAX_BODY_BYTES: usize = 1 << 20;

/// A parsed HTTP response: the numeric status + the body string (decoded lossy so
/// a stray byte can never panic — the same Chinese-Windows discipline as the
/// bridge reader).
#[derive(Debug, Clone, PartialEq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

impl HttpResponse {
    /// `true` for a 2xx status (the only "ok" the pollers act on).
    pub fn is_ok(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// GET `http://<host>:<port><path>` (with the path already including any
/// `?query`). Returns `None` on ANY failure (connect refused, timeout, malformed
/// response) — the caller treats that as "server down". Bounded by [`HTTP_TIMEOUT`].
pub fn get(host: &str, port: u16, path: &str) -> Option<HttpResponse> {
    request(host, port, "GET", path, None)
}

/// POST a JSON body to `http://<host>:<port><path>`. Used for conductor node
/// actions (`POST /subagent/{id}`). Returns `None` on failure. Bounded.
pub fn post_json(host: &str, port: u16, path: &str, json_body: &str) -> Option<HttpResponse> {
    request(host, port, "POST", path, Some(json_body))
}

/// The shared request path: open a timed TcpStream, write the request bytes, read
/// the whole (bounded) response, and parse it. `body` is a JSON string for POST.
fn request(host: &str, port: u16, method: &str, path: &str, body: Option<&str>) -> Option<HttpResponse> {
    let addr = (host, port).to_socket_addrs().ok()?.next()?;
    let mut stream = TcpStream::connect_timeout(&addr, HTTP_TIMEOUT).ok()?;
    stream.set_read_timeout(Some(HTTP_TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(HTTP_TIMEOUT)).ok()?;

    let req = build_request(method, host, port, path, body);
    stream.write_all(req.as_bytes()).ok()?;
    stream.flush().ok()?;

    // Read the whole response (bounded). We close on EOF; `Connection: close`
    // (HTTP/1.0, no keep-alive) makes the server close the socket after the body.
    let mut raw: Vec<u8> = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&chunk[..n]);
                if raw.len() >= MAX_BODY_BYTES {
                    raw.truncate(MAX_BODY_BYTES);
                    break;
                }
            }
            Err(_) => break, // timeout / reset → use whatever we have.
        }
    }
    parse_response(&raw)
}

/// Build the raw HTTP/1.0 request string (PURE — unit-tested). HTTP/1.0 +
/// `Connection: close` so the server closes after the body and our read loop ends
/// on EOF (no chunked-encoding parsing needed for these tiny localhost replies).
pub fn build_request(method: &str, host: &str, port: u16, path: &str, body: Option<&str>) -> String {
    let mut req = format!(
        "{method} {path} HTTP/1.0\r\nHost: {host}:{port}\r\nAccept: application/json\r\nConnection: close\r\n"
    );
    if let Some(b) = body {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
        req.push_str("\r\n");
        req.push_str(b);
    } else {
        req.push_str("\r\n");
    }
    req
}

/// Parse raw response bytes into [`HttpResponse`] (PURE — unit-tested). Splits the
/// status line + headers from the body at the first `\r\n\r\n`, reads the numeric
/// status, and decodes the body `from_utf8_lossy`. Returns `None` if there is no
/// recognizable status line.
pub fn parse_response(raw: &[u8]) -> Option<HttpResponse> {
    // Find the header/body separator.
    let sep = find_subslice(raw, b"\r\n\r\n");
    let (head, body_bytes) = match sep {
        Some(i) => (&raw[..i], &raw[i + 4..]),
        // Some minimal servers may use bare \n\n; tolerate it.
        None => match find_subslice(raw, b"\n\n") {
            Some(i) => (&raw[..i], &raw[i + 2..]),
            None => (raw, &b""[..]),
        },
    };
    let head_str = String::from_utf8_lossy(head);
    let status_line = head_str.lines().next()?;
    let status = parse_status_code(status_line)?;
    let body = String::from_utf8_lossy(body_bytes).trim().to_string();
    Some(HttpResponse { status, body })
}

/// Extract the numeric status code from an HTTP status line
/// (`HTTP/1.1 200 OK` → `200`). PURE.
fn parse_status_code(status_line: &str) -> Option<u16> {
    let mut parts = status_line.split_whitespace();
    let _http = parts.next()?; // "HTTP/1.x"
    let code = parts.next()?; // "200"
    code.parse::<u16>().ok()
}

/// First index of `needle` in `haystack` (a tiny substring search; PURE).
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_get_is_http10_close() {
        let req = build_request("GET", "127.0.0.1", 8900, "/subagent", None);
        assert!(req.starts_with("GET /subagent HTTP/1.0\r\n"));
        assert!(req.contains("Host: 127.0.0.1:8900\r\n"));
        assert!(req.contains("Connection: close\r\n"));
        assert!(req.ends_with("\r\n\r\n"), "no-body request ends with a blank line");
        assert!(!req.contains("Content-Length"));
    }

    #[test]
    fn build_request_post_has_body_and_length() {
        let body = r#"{"action":"stop"}"#;
        let req = build_request("POST", "127.0.0.1", 8900, "/subagent/ab12", Some(body));
        assert!(req.starts_with("POST /subagent/ab12 HTTP/1.0\r\n"));
        assert!(req.contains("Content-Type: application/json\r\n"));
        assert!(req.contains(&format!("Content-Length: {}\r\n", body.len())));
        assert!(req.ends_with(body), "the JSON body is appended after the headers");
    }

    #[test]
    fn parse_response_splits_status_and_body() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"items\":[]}";
        let resp = parse_response(raw).expect("parses");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, r#"{"items":[]}"#);
        assert!(resp.is_ok());

        // A 404 (e.g. bad board key) parses with the right status + a non-2xx.
        let nf = parse_response(b"HTTP/1.0 404 Not Found\r\n\r\nNot Found").expect("parses 404");
        assert_eq!(nf.status, 404);
        assert!(!nf.is_ok());

        // Bare \n\n separator is tolerated.
        let bare = parse_response(b"HTTP/1.1 200 OK\n\nhi").expect("bare-nn parses");
        assert_eq!(bare.status, 200);
        assert_eq!(bare.body, "hi");

        // No status line → None (never panics).
        assert!(parse_response(b"garbage with no http line").is_none());
        assert!(parse_response(b"").is_none());
    }

    #[test]
    fn find_subslice_locates_separator() {
        assert_eq!(find_subslice(b"abc\r\n\r\ndef", b"\r\n\r\n"), Some(3));
        assert_eq!(find_subslice(b"abcdef", b"xyz"), None);
        assert_eq!(find_subslice(b"", b"x"), None);
    }

    /// A GET against a port that is (almost certainly) closed returns None FAST —
    /// proving the watcher's "server down" path fails fast and never hangs the
    /// thread. (Uses a high port unlikely to be bound; if it happens to be bound on
    /// the gate machine, a non-None ok result is also acceptable — the assertion is
    /// only that it RETURNS within the timeout, which the test harness enforces.)
    #[test]
    fn get_on_closed_port_returns_fast() {
        let start = std::time::Instant::now();
        let _ = get("127.0.0.1", 59321, "/subagent");
        // Must return within a small multiple of the timeout (it should be ~instant
        // on connection-refused; the timeout only applies if something half-opens).
        assert!(
            start.elapsed() < HTTP_TIMEOUT * 4,
            "a closed-port GET must fail fast, took {:?}",
            start.elapsed()
        );
    }
}
