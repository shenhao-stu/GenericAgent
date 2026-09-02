# Desktop packaging

This directory contains the scripts and locked Python inputs used to assemble GenericAgent Desktop packages.
The directory is not copied wholesale into a release; the workflow selects only the platform scripts and
requirements files required by each package.

## Renderer build boundary

Package jobs build the renderer from the in-tree source (`frontends/desktop/src/**`): `npm ci`, `npm run typecheck`,
`npm test`, then `npm run tauri build`, whose `beforeBuildCommand` runs `npm run build:tauri-assets` (Vite build,
`dist/build-provenance.json`, and the `test:dist` contract) before Tauri embeds `dist/**` in the native application.
`dist/**` is never tracked. The portable `runtime/app` copy deliberately excludes `dist/**`, `src/**`, `public/**`,
`node_modules/**` and the renderer tooling so the renderer is not duplicated as loose runtime data.

`frontends/desktop/static/**` remains the independent Desktop v1 and is unchanged by Desktop 2.0 packaging.

The package, bundle, qualification, and provenance version for this release is `0.2.3`.

The Semi Design MIT text is tracked as `public/THIRD_PARTY_NOTICES.txt` (copied to `dist/` by the build). Root
attributes force it to LF on every platform, and `test:dist` verifies the locked notice SHA-256 in the built tree.

## Inputs

```text
frontends/desktop/packaging/
├── README.md
├── dmg-build-requirements.txt
├── python-runtime-requirements.txt
└── scripts/
    ├── merge_desktop_settings.py
    ├── windows/
    │   ├── install_windows.ps1
    │   ├── uninstall.bat
    │   └── uninstall_windows.ps1
    ├── linux/
    │   ├── install_linux.sh
    │   └── uninstall.sh
    └── macos/
        ├── install_macos.sh
        └── uninstall.command
```

- Node is fixed at `22.23.2`; builders install the renderer toolchain from `package-lock.json` with `npm ci`
  (Tauri CLI `2.11.4` included). Rust is fixed at `1.95.0` and crates are locked by `Cargo.lock`.
- Root `.gitattributes` pins `public/THIRD_PARTY_NOTICES.txt` to LF. The dist contract runs inside `tauri build`
  immediately before Tauri embeds `dist`, locking the notice SHA-256 to
  `2acf865e87e59090121369aac0575467067fdd7999923a70d785a46ceae3330f` and failing on any byte drift.
- python-build-standalone is fixed at release `20260814`, CPython `3.12.14`, an explicit architecture, and an
  archive SHA-256 on each platform.
- The macOS builder runs on the Apple-silicon `macos-26` host with
  `DEVELOPER_DIR=/Applications/Xcode_26.5.app/Contents/Developer` and hard-fails unless Xcode `26.5`
  (build `17F42`) and the macOS `26.5` SDK are active. Host CPython `3.12.10` is used only for the hash-locked
  DMG layout tools and remains separate from the embedded `3.12.14` runtime.
- Windows converts `RUNNER_TEMP` with `cygpath -u` before passing its archive path to POSIX tools.
- Linux builds on Ubuntu 22.04, uses a dedicated glibc-2.35 Rust cache prefix with target/bin caching disabled,
  and scans every packaged ELF to reject a maximum GLIBC requirement above 2.35.
- Runtime direct and transitive dependencies use exact versions in `python-runtime-requirements.txt`; downloads
  require binary wheels. The DMG layout requirements use exact versions and wheel hashes.
- Preparation disables Python bytecode writes and removes `__pycache__`, `.pyc`, and `.pyo` files across the
  complete bundled runtime without removing vendored Python packages.
- Checkout, Node/Python setup, artifact upload/download, Rust toolchain, and Rust cache actions use full commit
  SHA pins and disable persisted checkout credentials.

These controls do not make the packages bit-for-bit reproducible. Runner images, Ubuntu packages, the Windows
SDK/UCRT input, platform wheels, and native packaging metadata can change or contain timestamps. Candidate
artifact SHA-256 values and release-qualification reports remain the evidence for a particular build.

## Candidate artifacts and publication

Manual `workflow_dispatch` runs build the selected platform or all three platforms as downloadable Actions
artifacts. They never create or edit a GitHub Release.

A maintainer-created `desktop-portable-*` tag starts this publication graph:

```text
Windows builder (contents: read) ─┐
Linux builder   (contents: read) ─┼─ verified Actions artifacts ─┐
macOS builder   (contents: read) ─┘                              │
                                                                  ▼
                                           publisher (contents: write)
                                           validate six files/checksums
                                           create one invisible draft
                                           verify uploaded asset set
                                           expose one prerelease
```

The three builders do not receive a write token and do not call the GitHub Release API. The publisher runs only
for a matching tag push after all three builders succeed; it does not check out source or run package tooling.
It refuses to overwrite an existing Release for the tag.

The prerelease contains exactly:

- `GenericAgent-Desktop-Windows-Portable.zip`
- `SHA256SUMS-windows.txt`
- `GenericAgent-Desktop-Linux-Portable.tar.gz`
- `SHA256SUMS-linux.txt`
- `GenericAgent-Desktop-macOS-aarch64.dmg`
- `GenericAgent-Desktop-macOS-aarch64.dmg.sha256`

The macOS application is ad-hoc signed only. It is not Developer ID signed or notarized, and the published notes
retain the documented context-menu / Privacy & Security opening path.

## Validation

- `npm run test:dist` verifies required built entries, references, source leakage, dead assets, the notice hash,
  version/CSP/E2E isolation, the build provenance manifest, and the atomic release contract.
- `python -m pytest frontends/tests` verifies Desktop bridge, conductor, model/session/upload behavior, the
  Desktop-only cost ledger, transactional data import, release-qualification tooling, owned service/port
  behavior, `GA_ROOT`, and Tauri security contracts.
- Rust formatting, clippy, production/E2E tests, and a Tauri `--no-bundle` build validate the native shell against
  the freshly built renderer.
- Shell, Python, PowerShell, YAML, action pinning, boundary diff, and conflict checks run before delivery.
