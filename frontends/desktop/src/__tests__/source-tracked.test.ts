// @vitest-environment node
import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * Every relative import in the renderer must resolve to a git-tracked file with byte-exact casing.
 * Windows resolves imports case-insensitively and the repo's root .gitignore is aggressive (`.*`, `**log.*`),
 * so an untracked or wrong-case file only surfaces as a Linux CI build failure otherwise.
 */
const SRC = path.resolve(__dirname, '..');
const DESKTOP = path.resolve(SRC, '..');
const RESOLVE_EXTS = ['', '.ts', '.tsx', '.js', '.mjs', '.css', '.json', '/index.ts', '/index.tsx'];
const IMPORT_RE = /(?:from\s+|import\s+|import\()\s*['"](\.{1,2}\/[^'"]+)['"]/g;
const CSS_IMPORT_RE = /@import\s+(?:url\()?['"](\.{1,2}\/[^'"]+)['"]/g;

function tracked(): Set<string> {
  const out = execFileSync('git', ['ls-files', '-z'], { cwd: DESKTOP }).toString('utf8');
  return new Set(out.split('\0').filter(Boolean).map((p) => path.resolve(DESKTOP, p)));
}

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    return statSync(full).isDirectory() ? walk(full) : /\.(tsx?|css)$/.test(name) ? [full] : [];
  });
}

function resolveImport(from: string, spec: string, files: Set<string>): string | null {
  const base = path.resolve(path.dirname(from), spec);
  return RESOLVE_EXTS.map((ext) => path.normalize(base + ext)).find((candidate) => files.has(candidate)) ?? null;
}

describe('renderer source integrity', () => {
  it('every relative import resolves to a git-tracked file with exact casing', () => {
    const files = tracked();
    const missing: string[] = [];
    for (const file of walk(SRC)) {
      const text = readFileSync(file, 'utf8');
      const re = file.endsWith('.css') ? CSS_IMPORT_RE : IMPORT_RE;
      for (const [, spec] of text.matchAll(re)) {
        if (!resolveImport(file, spec, files)) missing.push(`${path.relative(SRC, file)} -> ${spec}`);
      }
    }
    expect(missing).toEqual([]);
  });
});
