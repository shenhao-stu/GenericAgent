// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { zh } from '../i18n/zh';
import { en } from '../i18n/en';
import { t } from '../i18n/t';

const placeholders = (text: string) => [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

describe('i18n dictionaries', () => {
  it('zh and en expose exactly the same keys', () => {
    const zhKeys = Object.keys(zh).sort();
    const enKeys = Object.keys(en).sort();
    expect(enKeys.filter((k) => !(k in zh))).toEqual([]);
    expect(zhKeys.filter((k) => !(k in en))).toEqual([]);
  });

  it('every key carries the same interpolation placeholders in both languages', () => {
    const mismatched = Object.keys(zh).filter((k) => placeholders(zh[k]).join() !== placeholders(en[k] ?? '').join());
    expect(mismatched).toEqual([]);
  });

  it('no dictionary value is empty', () => {
    const empties = [...Object.entries(zh), ...Object.entries(en)].filter(([, v]) => !v.trim()).map(([k]) => k);
    expect(empties).toEqual([]);
  });

  it('t() interpolates params and falls back to the key for unknown lookups', () => {
    expect(t('en', 'fold.chars', { n: 42 })).toBe('42 chars');
    expect(t('zh', 'fold.turn', { n: 3 })).toBe('第 3 轮');
    expect(t('en', 'does.not.exist')).toBe('does.not.exist');
    expect(t('fr', 'common.close')).toBe(zh['common.close']);
  });
});
