// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { PROVIDER_PRESETS, presetColor, presetIcon, presetLabel, presetsForLang } from '../data/model-presets';
import { en } from '../i18n/en';
import { zh } from '../i18n/zh';

describe('provider presets', () => {
  it('every preset is fully describable in both languages and has a registered icon', () => {
    for (const p of PROVIDER_PRESETS) {
      expect(zh[p.descKey], p.key).toBeTruthy();
      expect(en[p.descKey], p.key).toBeTruthy();
      expect(presetLabel(p, 'zh')).toBeTruthy();
      expect(presetLabel(p, 'en')).toMatch(/^[\x20-\x7e]+$/); // English labels never leak CJK brand names
      expect(presetIcon(p)).toBeTruthy();
      expect(presetColor(p)).toMatch(/^#/);
      expect(p.apibase).toMatch(/^https?:\/\//);
      expect(p.keyUrl).toMatch(/^https:\/\//);
    }
    expect(new Set(PROVIDER_PRESETS.map((p) => p.key)).size).toBe(PROVIDER_PRESETS.length);
  });

  it('orders global providers first for English and keeps the Chinese order otherwise', () => {
    expect(presetsForLang('en').slice(0, 3).map((p) => p.key)).toEqual(['openai', 'anthropic', 'google']);
    expect(presetsForLang('zh')).toBe(PROVIDER_PRESETS);
    expect(presetsForLang('en')).toHaveLength(PROVIDER_PRESETS.length);
  });

  it('a local runtime preset carries a placeholder key so the bridge invariant (non-empty key) holds', () => {
    expect(PROVIDER_PRESETS.find((p) => p.key === 'ollama')?.defaultKey).toBeTruthy();
    expect(PROVIDER_PRESETS.filter((p) => p.key !== 'ollama').every((p) => p.defaultKey === undefined)).toBe(true);
  });
});
