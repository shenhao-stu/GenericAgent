// Contract check for the built renderer (dist/**) and the files that package it.
// dist/ is a build artifact: `npm run build` produces it together with build-provenance.json.
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { checkDistBuilt } from './assert-dist-built.mjs';
import { PROVENANCE_FILE, desktopDir, distDir, distManifest, repoRoot, sha256 } from './dist-manifest.mjs';
import { SEMI_UI_NOTICE_CONTRACT } from './react-public-assets.mjs';

const noticeAttributeRule = 'frontends/desktop/public/THIRD_PARTY_NOTICES.txt text eol=lf';
const releaseAssets = [
  'GenericAgent-Desktop-Windows-Portable.zip',
  'SHA256SUMS-windows.txt',
  'GenericAgent-Desktop-Linux-Portable.tar.gz',
  'SHA256SUMS-linux.txt',
  'GenericAgent-Desktop-macOS-aarch64.dmg',
  'GenericAgent-Desktop-macOS-aarch64.dmg.sha256',
];
const builderBundles = { 'build-windows': 'nsis', 'build-linux': 'appimage', 'build-macos': 'app' };

function fail(message) {
  throw new Error(message);
}

async function exists(file) {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}

async function readJson(file) {
  return JSON.parse(await readFile(file, 'utf8'));
}

function workflowJob(workflow, name) {
  const match = new RegExp(`^  ${name}:\\s*$`, 'm').exec(workflow);
  if (!match) return '';
  const remainder = workflow.slice(match.index + match[0].length);
  const nextJob = /^  [a-zA-Z0-9_-]+:\s*$/m.exec(remainder);
  return nextJob ? remainder.slice(0, nextJob.index) : remainder;
}

async function verifyHtmlReferences(relativePath) {
  const html = await readFile(path.join(distDir, relativePath), 'utf8');
  for (const [, reference] of html.matchAll(/\b(?:src|href)=["']([^"']+)["']/g)) {
    if (/^(?:[a-z]+:|#|\/\/)/i.test(reference)) continue;
    const clean = reference.split(/[?#]/, 1)[0].replace(/^\//, '');
    if (clean && !await exists(path.join(distDir, clean))) fail(`${relativePath} references a missing asset: ${clean}`);
  }
}

async function verifyBuiltTree() {
  const built = checkDistBuilt(distDir);
  if (!built.ok) fail(built.error);
  const files = (await distManifest()).files.map(({ relativePath }) => relativePath);
  const leaked = files.filter((file) => /(?:\.map|\.tsx?|\.jsx)$/i.test(file));
  if (leaked.length > 0) fail(`dist contains source files: ${leaked.join(', ')}`);
  for (const file of files.filter((file) => /\.(?:html|js|css)$/i.test(file))) {
    const contents = await readFile(path.join(distDir, file), 'utf8');
    if (/(?:webdriverio|__GA_E2E__|wdio:|sourceMappingURL|webpack:\/\/|\/Users\/)/i.test(contents)) {
      fail(`dist leaks source/E2E material: ${file}`);
    }
  }
  const notice = await readFile(path.join(distDir, SEMI_UI_NOTICE_CONTRACT.publicAsset));
  if (sha256(notice) !== SEMI_UI_NOTICE_CONTRACT.sha256) fail('Semi Design third-party notice hash is incorrect');
  await Promise.all(['index.html', 'loading.html', 'setup.html'].map(verifyHtmlReferences));
}

async function verifyProvenance(version) {
  const file = path.join(distDir, PROVENANCE_FILE);
  if (!await exists(file)) fail(`dist/${PROVENANCE_FILE} is missing — run \`npm run build\``);
  const provenance = await readJson(file);
  const manifest = await distManifest();
  if (provenance.version !== version) fail(`provenance version ${provenance.version} != package version ${version}`);
  if (provenance.generatedAssetCount !== manifest.count) {
    fail(`provenance asset count mismatch: expected ${provenance.generatedAssetCount}, found ${manifest.count}`);
  }
  if (provenance.generatedManifestSha256 !== manifest.digest) {
    fail('dist/** does not match its provenance manifest — rebuild instead of editing generated files');
  }
}

async function verifyNoticeAttribute() {
  const lines = (await readFile(path.join(repoRoot, '.gitattributes'), 'utf8')).split(/\r?\n/);
  const noticePath = noticeAttributeRule.split(/\s+/, 1)[0];
  const rules = lines.filter((line) => line.trim().split(/\s+/, 1)[0] === noticePath);
  if (rules.length !== 1 || rules[0] !== noticeAttributeRule) {
    fail(`third-party notice must have one exact LF attribute rule: ${noticeAttributeRule}`);
  }
}

async function verifyVersionsAndShell(version) {
  const tauriConfig = await readJson(path.join(desktopDir, 'src-tauri', 'tauri.conf.json'));
  const tauriE2eConfig = await readJson(path.join(desktopDir, 'src-tauri', 'tauri.e2e.conf.json'));
  const cargoToml = await readFile(path.join(desktopDir, 'src-tauri', 'Cargo.toml'), 'utf8');
  const cargoLock = await readFile(path.join(desktopDir, 'src-tauri', 'Cargo.lock'), 'utf8');
  const shell = await readFile(path.join(desktopDir, 'src-tauri', 'src', 'lib.rs'), 'utf8');
  const escaped = version.replace(/\./g, '\\.');

  if (tauriConfig.version !== version) fail(`tauri.conf.json version must be ${version}`);
  if (tauriConfig.build?.frontendDist !== '../dist') fail('Tauri frontendDist must be ../dist');
  if (tauriConfig.build?.beforeBuildCommand !== 'npm run build:tauri-assets') {
    fail('Tauri beforeBuildCommand must build and verify the renderer');
  }
  if (!new RegExp(`^\\[package\\][\\s\\S]*?^name = "ga-desktop"$[\\s\\S]*?^version = "${escaped}"$`, 'm').test(cargoToml)) {
    fail(`Cargo.toml ga-desktop version must be ${version}`);
  }
  if (!new RegExp(`^\\[\\[package\\]\\][\\s\\S]*?^name = "ga-desktop"$\\nversion = "${escaped}"$`, 'm').test(cargoLock)) {
    fail(`Cargo.lock ga-desktop version must be ${version}`);
  }

  const security = tauriConfig.app?.security;
  if (!security?.csp || typeof security.csp !== 'object') fail('production renderer must use an explicit CSP');
  if (String(security.csp['script-src']).includes('unsafe-eval')) fail('production CSP must not enable unsafe-eval');
  if (!String(security.csp['connect-src']).includes('127.0.0.1:14168')) {
    fail('production CSP must retain the Desktop loopback bridge');
  }
  if (JSON.stringify(security).includes('wdio:')) fail('production Tauri security config must not grant WebDriver');
  if (!JSON.stringify(tauriE2eConfig).includes('wdio:default')) {
    fail('E2E-only Tauri config must retain its isolated WebDriver capability');
  }
  if (!shell.includes('fn main_ui_url_from_current')) {
    fail('desktop shell must resolve index.html from the active Tauri asset origin');
  }
  if (/tauri::Url::parse\("http:\/\/127\.0\.0\.1:14168\/?"\)/.test(shell)) {
    fail('desktop shell must not navigate the renderer to the legacy bridge root');
  }
}

async function verifyReleaseContract() {
  const workflow = await readFile(path.join(repoRoot, '.github', 'workflows', 'desktop-release-package.yml'), 'utf8');
  const buildJobs = Object.keys(builderBundles);
  const jobNames = [...workflow.slice(workflow.search(/^jobs:\s*$/m)).matchAll(/^  ([a-zA-Z0-9_-]+):\s*$/gm)]
    .map((match) => match[1]);
  if (JSON.stringify(jobNames) !== JSON.stringify([...buildJobs, 'publish-release'])) {
    fail('release workflow must contain exactly three builders and one publisher');
  }
  if (!/^permissions:\n  contents: read\s*$/m.test(workflow)) fail('release workflow must default to contents: read');
  if ((workflow.match(/^      contents: write\s*$/gm) ?? []).length !== 1) {
    fail('release workflow must grant contents: write exactly once');
  }

  for (const job of buildJobs) {
    const body = workflowJob(workflow, job);
    if (!body) fail(`missing release builder: ${job}`);
    if (!/^    permissions:\n      contents: read\s*$/m.test(body) || body.includes('contents: write')) {
      fail(`${job} must have read-only repository permission`);
    }
    if (!/uses: actions\/checkout@[0-9a-f]{40}[^\n]*\n        with:\n          persist-credentials: false/.test(body)) {
      fail(`${job} checkout must use a full SHA and disable persisted credentials`);
    }
    if (body.includes('secrets.GITHUB_TOKEN') || /\bgh release\b/.test(body)) {
      fail(`${job} must not receive a release token or publish`);
    }
    if (!/^        run: npm ci$/m.test(body)) fail(`${job} must install renderer dependencies from the lockfile`);
    if (!body.includes('npm run typecheck') || !body.includes('npm test')) {
      fail(`${job} must typecheck and unit-test the renderer before packaging`);
    }
    if (!body.includes(`npm run tauri build -- --bundles ${builderBundles[job]}`)) {
      fail(`${job} must build the ${builderBundles[job]} bundle through the Tauri CLI`);
    }
    if (!/--exclude='(?:\.\/)?frontends\/desktop\/dist'/.test(body)) {
      fail(`${job} must not duplicate Tauri-embedded dist inside runtime/app`);
    }
    if (!/--exclude='(?:\.\/)?frontends\/desktop\/(?:release_qualification|node_modules)'/.test(body)) {
      fail(`${job} must not package renderer tooling inside runtime/app`);
    }
    if (!/test ! -e "\$RUNTIME(?:_SRC)?\/app\/frontends\/desktop\/dist"/.test(body)) {
      fail(`${job} must verify dist is not duplicated inside runtime/app`);
    }
  }

  const windows = workflowJob(workflow, 'build-windows');
  if (!windows.includes('cygpath -u "$RUNNER_TEMP"')) {
    fail('Windows packaging must convert RUNNER_TEMP with cygpath before POSIX tools use it');
  }
  const linux = workflowJob(workflow, 'build-linux');
  if (!linux.includes('runs-on: ubuntu-22.04') || !linux.includes('cache-targets: "false"')) {
    fail('Linux packaging must isolate its Ubuntu 22.04/glibc 2.35 Rust cache ABI');
  }

  const publisher = workflowJob(workflow, 'publish-release');
  if (!publisher.includes('needs: [build-windows, build-linux, build-macos]')
      || !publisher.includes("github.event_name == 'push'")
      || !publisher.includes('refs/tags/desktop-portable-')
      || !/^    permissions:\n      contents: write\s*$/m.test(publisher)) {
    fail('publisher must be the sole tag-only writer after all three builders');
  }
  const publisherRun = publisher.slice(publisher.indexOf('        run: |'));
  if (!publisher.includes('TAG_NAME: ${{ github.ref_name }}')
      || publisherRun.includes('${{ github.ref_name }}')
      || !publisherRun.includes('^desktop-portable-[A-Za-z0-9._-]+$')) {
    fail('publisher must pass the tag through env and validate it before shell use');
  }
  if ((publisher.match(/actions\/download-artifact@/g) ?? []).length !== 3
      || (publisher.match(/\bgh release create\b/g) ?? []).length !== 1
      || publisher.includes('gh release upload')
      || !publisher.includes('--draft')
      || !publisher.includes('gh release edit "$TAG_NAME" --draft=false --prerelease')) {
    fail('publisher must aggregate three artifacts as a verified draft, then expose one prerelease');
  }
  for (const file of releaseAssets) {
    if (!publisher.includes(file)) fail(`publisher does not require release asset: ${file}`);
  }
}

async function main() {
  const { version } = await readJson(path.join(desktopDir, 'package.json'));
  await verifyBuiltTree();
  await verifyProvenance(version);
  await verifyNoticeAttribute();
  await verifyVersionsAndShell(version);
  await verifyReleaseContract();
  console.log(`Desktop renderer ${version}: dist contracts passed.`);
}

await main();
