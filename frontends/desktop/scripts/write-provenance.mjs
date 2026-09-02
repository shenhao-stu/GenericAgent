// Runs after `vite build`: records which source produced dist/** so verify-compiled-dist.mjs can
// prove the packaged renderer is exactly this build (not a stale or hand-edited tree).
import { execFileSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { PROVENANCE_FILE, desktopDir, distDir, distManifest, repoRoot } from './dist-manifest.mjs';

function git(...args) {
  try {
    return execFileSync('git', args, { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return '';
  }
}

const pkg = JSON.parse(await readFile(path.join(desktopDir, 'package.json'), 'utf8'));
const manifest = await distManifest();
const provenance = {
  schemaVersion: 2,
  product: 'GenericAgent Desktop renderer',
  version: pkg.version,
  sourceCommit: git('rev-parse', 'HEAD') || 'unknown',
  sourceDirty: git('status', '--porcelain', '--', 'frontends/desktop') !== '',
  generatedAssetCount: manifest.count,
  generatedManifestSha256: manifest.digest,
};
await writeFile(path.join(distDir, PROVENANCE_FILE), `${JSON.stringify(provenance, null, 2)}\n`);
console.log(`wrote dist/${PROVENANCE_FILE} (${manifest.count} assets, ${provenance.sourceCommit.slice(0, 7)})`);
