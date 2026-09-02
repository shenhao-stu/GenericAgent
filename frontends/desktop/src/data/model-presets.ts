import type { Lang } from '../i18n/t';
import { PROVIDER_ICONS } from './provider-icons';

/**
 * One row per provider the add-model picker can pre-fill. Icons and brand colours come from the shared
 * provider registry; the only per-language data is the display name (Chinese brands read differently in
 * each language), everything else is protocol facts.
 */
export interface ProviderPreset {
  key: keyof typeof PROVIDER_ICONS;
  label: Record<Lang, string>;
  descKey: string;
  protocol: 'oai' | 'claude';
  apibase: string;
  defaultModel: string;
  /** Pre-filled API key for backends that ignore it (local runtimes); the bridge still requires a non-empty key. */
  defaultKey?: string;
  keyUrl: string;
}

const preset = (
  key: ProviderPreset['key'],
  label: string | Record<Lang, string>,
  protocol: ProviderPreset['protocol'],
  apibase: string,
  defaultModel: string,
  keyUrl: string,
  extra: Partial<ProviderPreset> = {},
): ProviderPreset => ({
  key,
  label: typeof label === 'string' ? { zh: label, en: label } : label,
  descKey: `pq.${key}Desc`,
  protocol,
  apibase,
  defaultModel,
  keyUrl,
  ...extra,
});

export const PROVIDER_PRESETS: ProviderPreset[] = [
  preset('deepseek', 'DeepSeek', 'oai', 'https://api.deepseek.com/v1', 'deepseek-v4-pro', 'https://platform.deepseek.com/api_keys'),
  preset('qwen', { zh: '通义千问', en: 'Qwen' }, 'oai', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'qwen3.7-max', 'https://bailian.console.aliyun.com/?apiKey=1'),
  preset('doubao', { zh: '火山方舟', en: 'Volcengine Ark' }, 'oai', 'https://ark.cn-beijing.volces.com/api/v3', 'doubao-seed-2.1-pro', 'https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey'),
  preset('zhipu', { zh: '智谱 GLM', en: 'Zhipu GLM' }, 'oai', 'https://open.bigmodel.cn/api/paas/v4', 'glm-5.2', 'https://open.bigmodel.cn/usercenter/apikeys'),
  preset('kimi', 'Kimi', 'oai', 'https://api.moonshot.cn/v1', 'kimi-k2.7-code', 'https://platform.moonshot.cn/console/api-keys'),
  preset('minimax', 'MiniMax', 'oai', 'https://api.minimax.chat/v1', 'MiniMax-M3', 'https://platform.minimaxi.com/user-center/basic-information/interface-key'),
  preset('stepfun', { zh: '阶跃星辰', en: 'StepFun' }, 'oai', 'https://api.stepfun.com/v1', 'step-3.7-flash', 'https://platform.stepfun.com/interface-key'),
  preset('openai', 'OpenAI', 'oai', 'https://api.openai.com/v1', 'gpt-5.4', 'https://platform.openai.com/api-keys'),
  preset('anthropic', 'Anthropic', 'claude', 'https://api.anthropic.com', 'claude-sonnet-4-6', 'https://console.anthropic.com/settings/keys'),
  preset('google', 'Google Gemini', 'oai', 'https://generativelanguage.googleapis.com/v1beta/openai', 'gemini-2.5-pro', 'https://aistudio.google.com/apikey'),
  preset('openrouter', 'OpenRouter', 'oai', 'https://openrouter.ai/api/v1', 'openrouter/auto', 'https://openrouter.ai/settings/keys'),
  preset('xai', 'xAI Grok', 'oai', 'https://api.x.ai/v1', 'grok-4', 'https://console.x.ai'),
  preset('ollama', 'Ollama', 'oai', 'http://127.0.0.1:11434/v1', 'llama3.1', 'https://ollama.com/download', { defaultKey: 'ollama' }),
];

/** Chinese-first for zh, global-first for en: the first screen should show the providers a user is likely to hold a key for. */
const GLOBAL_FIRST: ProviderPreset['key'][] = ['openai', 'anthropic', 'google', 'deepseek', 'openrouter', 'xai', 'ollama'];

export function presetsForLang(lang: Lang): ProviderPreset[] {
  if (lang !== 'en') return PROVIDER_PRESETS;
  const rank = (p: ProviderPreset) => { const i = GLOBAL_FIRST.indexOf(p.key); return i < 0 ? GLOBAL_FIRST.length : i; };
  return [...PROVIDER_PRESETS].sort((a, b) => rank(a) - rank(b));
}

export const presetLabel = (p: ProviderPreset, lang: Lang) => p.label[lang];
export const presetColor = (p: ProviderPreset) => PROVIDER_ICONS[p.key].color;
export const presetIcon = (p: ProviderPreset) => PROVIDER_ICONS[p.key].Component;
