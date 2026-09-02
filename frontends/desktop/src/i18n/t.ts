import { zh } from './zh';
import { en } from './en';

export type Lang = 'zh' | 'en';

const dictionaries: Record<Lang, Record<string, string>> = { zh, en };

/** Pure lookup with `{param}` interpolation; unknown keys return the key itself. */
export function t(lang: string, key: string, params?: Record<string, string | number>): string {
  const dict = dictionaries[lang as Lang] || dictionaries.zh;
  let text = dict[key] ?? key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      text = text.replaceAll(`{${k}}`, String(v));
    }
  }
  return text;
}

/** Language chosen before React mounts (index.html boot script / navigator), for store-free entries. */
export function bootLang(): Lang {
  const html = document.documentElement.lang;
  if (html) return html === 'en' ? 'en' : 'zh';
  return (navigator.language || '').toLowerCase().startsWith('zh') ? 'zh' : 'en';
}
