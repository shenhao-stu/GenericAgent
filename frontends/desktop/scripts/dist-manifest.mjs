import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const desktopDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const repoRoot = path.resolve(desktopDir, '..', '..');
export const distDir = path.join(desktopDir, 'dist');
export const PROVENANCE_FILE = 'build-provenance.json';

export function sha256(contents) {
  return createHash('sha256').update(contents).digest('hex');
}

export async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walk(fullPath));
    if (entry.isFile()) files.push(fullPath);
  }
  return files;
}

export function relativeToDist(file) {
  return path.relative(distDir, file).split(path.sep).join('/');
}

/** Deterministic content manifest of dist/** (excluding the provenance file itself). */
export async function distManifest() {
  const files = (await walk(distDir))
    .map((file) => ({ file, relativePath: relativeToDist(file) }))
    .filter(({ relativePath }) => relativePath !== PROVENANCE_FILE)
    .sort((left, right) => left.relativePath.localeCompare(right.relativePath, 'en'));
  const lines = [];
  for (const { file, relativePath } of files) {
    lines.push(`${sha256(await readFile(file))}  ${relativePath}\n`);
  }
  return { files, count: files.length, digest: sha256(lines.join('')) };
}
