//! clipboard.rs — the copy/paste surface (OSC-52 → native), split out of `main.rs`
//! (ARCH Fix B). `perform_copy` is the single sink the `AppEvent::Copy` intent
//! resolves to; `export_action` performs the `/export` row choices. Every copy
//! outcome is surfaced as a notice (N1: never silent — the missing feedback was
//! the actual "can't copy" bug).

use crate::app::AppState;
use crate::i18n;
use crate::render;

/// Copy `text` via the clean logical-source path (P2): OSC 52 → native. RETURNS
/// the [`CopyResult`](render::copy::CopyResult) so callers can surface a visible
/// "Copied N bytes" notice — the result used to be swallowed, so a copy gave no
/// feedback at all (the root of the "can't copy" complaint).
pub(crate) fn copy_text(text: &str) -> render::copy::CopyResult {
    use render::copy::copy_to_clipboard;
    use render::{CopyCaps, Selection};
    use std::io::IsTerminal;
    let has_tty = std::io::stdout().is_terminal();
    copy_to_clipboard(text, Selection::Clipboard, CopyCaps::from_env(), has_tty)
}

/// Push a localized "Copied N bytes" (or an HONEST failure) notice for a finished
/// copy. `label` is an already-localized noun for what was copied ("selection",
/// "last reply", …). Every copy outcome is surfaced so the user always sees it
/// worked (N1: never silent — the missing feedback was the actual bug).
pub(crate) fn notice_copy(app: &mut AppState, res: &render::copy::CopyResult, label: &str) {
    use render::copy::CopyReason;
    if res.ok {
        app.push_notice(format!(
            "{} {label} · {} {}",
            i18n::t(app.lang, "copy.ok"),
            res.bytes,
            i18n::t(app.lang, "unit.bytes"),
        ));
    } else {
        let why = match res.reason {
            Some(CopyReason::TooLarge) => i18n::t(app.lang, "copy.fail.too_large"),
            Some(CopyReason::NoNativeRemote) | Some(CopyReason::NoTty) => {
                i18n::t(app.lang, "copy.fail.no_tty")
            }
            _ => i18n::t(app.lang, "copy.fail.generic"),
        };
        app.push_notice(format!("{} {label}: {why}", i18n::t(app.lang, "copy.fail")));
    }
}

/// Copy `text` and surface the outcome under `label` — the single sink the
/// `AppEvent::Copy` intent resolves to (the loop's `perform_actions` calls this).
pub(crate) fn perform_copy(app: &mut AppState, text: &str, label: &'static str) {
    let res = copy_text(text);
    notice_copy(app, &res, label);
}

/// Read the native clipboard for Ctrl+V (best-effort; `None` on any error).
pub(crate) fn read_clipboard() -> Option<String> {
    arboard::Clipboard::new().ok()?.get_text().ok()
}

/// Read a bitmap from the native clipboard (a copied screenshot) and re-encode
/// its raw RGBA pixels to PNG bytes, returning `(png_bytes, width, height)`.
/// Best-effort — `None` on any error (no image on the clipboard, the platform
/// backend lacks `image-data`, or an encode failure). This is the thin effectful
/// wrapper (OS clipboard read); the pure RGBA→PNG step is [`encode_rgba_png`],
/// which is unit-tested without a clipboard.
pub(crate) fn read_clipboard_image() -> Option<(Vec<u8>, u32, u32)> {
    let img = arboard::Clipboard::new().ok()?.get_image().ok()?;
    let w = u32::try_from(img.width).ok()?;
    let h = u32::try_from(img.height).ok()?;
    let png = encode_rgba_png(&img.bytes, w, h)?;
    Some((png, w, h))
}

/// Encode raw RGBA8 pixels (`width*height*4` bytes, row-major) to PNG bytes via
/// the `png` crate. Returns `None` on a size mismatch or an encoder error. PURE
/// over its inputs (no OS / clipboard), so it is unit-testable directly.
pub(crate) fn encode_rgba_png(rgba: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let expected = (width as usize).checked_mul(height as usize)?.checked_mul(4)?;
    if width == 0 || height == 0 || rgba.len() < expected {
        return None;
    }
    let mut out: Vec<u8> = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width, height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().ok()?;
        // arboard may hand back a longer buffer (stride padding); feed exactly the
        // `width*height*4` pixels the header declares.
        writer.write_image_data(&rgba[..expected]).ok()?;
        writer.finish().ok()?;
    }
    Some(out)
}

/// Perform an `/export` action by row id: 0 = clip (last reply via OSC52), 1 = all
/// (whole transcript to clipboard), 2 = file (last reply to a cwd file).
pub(crate) fn export_action(app: &mut AppState, id: usize) {
    match id {
        0 => {
            if let Some(src) = app.last_assistant_source() {
                let src = src.to_string();
                let label = i18n::t(app.lang, "copy.label.reply");
                let res = copy_text(&src);
                notice_copy(app, &res, label);
            } else {
                app.push_notice(i18n::tf(app.lang, "export.none"));
            }
        }
        1 => {
            let all = app.transcript_source();
            let label = i18n::t(app.lang, "copy.label.transcript");
            let res = copy_text(&all);
            notice_copy(app, &res, label);
        }
        2 => {
            let Some(src) = app.last_assistant_source().map(str::to_string) else {
                app.push_notice(i18n::tf(app.lang, "export.none"));
                return;
            };
            let fname = format!("tui_v4_export_{}.md", std::process::id());
            let path = app.repo_root.join(&fname);
            match std::fs::write(&path, &src) {
                Ok(()) => app.push_notice(format!("{} {}", i18n::t(app.lang, "export.wrote"), path.display())),
                Err(e) => app.push_notice(format!("{}: {e}", i18n::t(app.lang, "export.failed"))),
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S5 — the PURE RGBA→PNG encoder (`read_clipboard_image`'s testable half; the
    /// OS clipboard read itself is effectful + NOT unit-tested). A 2×2 RGBA buffer
    /// encodes to real PNG bytes (PNG magic header) that DECODE back to the same
    /// 2×2 RGBA via the `png` crate — proving the bytes are a valid image, not a
    /// blob. A size mismatch / zero dimension is rejected (`None`).
    #[test]
    fn encode_rgba_png_round_trips_a_valid_png() {
        // 2×2 RGBA: red, green / blue, white. (16 bytes = 2*2*4.)
        let rgba: Vec<u8> = vec![
            255, 0, 0, 255, 0, 255, 0, 255, // row 0
            0, 0, 255, 255, 255, 255, 255, 255, // row 1
        ];
        let png = encode_rgba_png(&rgba, 2, 2).expect("encode a 2x2 RGBA buffer");
        // PNG magic header.
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']);

        // Decode it back and confirm the dimensions + pixels survive.
        let decoder = png::Decoder::new(png.as_slice());
        let mut reader = decoder.read_info().expect("the encoded bytes are a valid PNG");
        let info = reader.info();
        assert_eq!((info.width, info.height), (2, 2));
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let frame = reader.next_frame(&mut buf).expect("decode frame");
        assert_eq!(&buf[..frame.buffer_size()], rgba.as_slice(), "pixels round-trip");

        // An over-long arboard buffer (stride padding) is truncated to w*h*4, OK.
        let mut padded = rgba.clone();
        padded.extend_from_slice(&[0, 0, 0, 0]);
        assert!(encode_rgba_png(&padded, 2, 2).is_some());

        // Too-small buffer / zero dimension → None (best-effort guard).
        assert!(encode_rgba_png(&rgba, 2, 3).is_none(), "fewer bytes than w*h*4 → None");
        assert!(encode_rgba_png(&rgba, 0, 2).is_none(), "zero width → None");
    }
}
