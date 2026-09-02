import { deleteJson, fetchJson, getJson, jsonInit, postJson, putJson } from './http';

export interface AppConfig {
  lang: 'zh' | 'en';
  theme: string;
  appearance: 'light' | 'dark';
  plain: boolean;
  fontSize: number;
  llmNo: number;
}

export interface ModelProfile {
  id: number;
  name: string;
  model: string;
  apibase: string;
  apikey?: string;
  protocol: 'oai' | 'claude';
  stream: boolean;
  max_retries?: number;
  connect_timeout?: number;
  read_timeout?: number;
  kind?: 'mixin';
  members?: string[];
  inMixin?: boolean;
}

export interface ModelProbeResult {
  ok: boolean;
  url?: string;
  status?: number;
  latencyMs?: number;
  error?: string;
}

type Profiles = { profiles: ModelProfile[] };
const profiles = (p: Promise<Profiles>) => p.then((r) => r.profiles ?? []);

const LOCAL_CONFIG_FALLBACK = (): AppConfig => ({
  lang: (localStorage.getItem('ga_lang') as 'zh' | 'en') || 'zh',
  theme: 'light',
  appearance: (localStorage.getItem('ga_appearance') as 'light' | 'dark') || 'light',
  plain: false,
  fontSize: parseInt(localStorage.getItem('ga_font_size') || '14', 10),
  llmNo: 0,
});

/** Preferences are also cached in localStorage by the settings store, so an offline bridge still boots the UI. */
export function getConfig(): Promise<AppConfig> {
  return getJson<{ config: AppConfig }>('/config').then((r) => r.config).catch(LOCAL_CONFIG_FALLBACK);
}

export function saveConfig(config: Partial<AppConfig>): Promise<void> {
  return postJson('/config', { config }).then(() => undefined);
}

export function getModelProfiles(): Promise<ModelProfile[]> {
  return profiles(getJson<Profiles>('/model-profiles')).catch(() => []);
}

export function getModelProfileDetail(id: number): Promise<ModelProfile | null> {
  return getJson<{ profile: ModelProfile }>(`/model-profiles/${id}`).then((r) => r.profile ?? null).catch(() => null);
}

export const addModelProfile = (data: Partial<ModelProfile>) => profiles(postJson<Profiles>('/model-profiles', data));
export const editModelProfile = (id: number, data: Partial<ModelProfile>) => profiles(putJson<Profiles>(`/model-profiles/${id}`, data));
export const deleteModelProfile = (id: number) => profiles(deleteJson<Profiles>(`/model-profiles/${id}`));
export const addToMixin = (id: number) => profiles(postJson<Profiles>(`/model-profiles/${id}/mixin`));
export const removeFromMixin = (id: number) => profiles(deleteJson<Profiles>(`/model-profiles/${id}/mixin`));
export const reorderMixin = (members: string[]) => profiles(putJson<Profiles>('/model-profiles/mixin/order', { members }));

/** One minimal round-trip against the provider described by `data`; nothing is persisted. A failed probe is a
 *  200 result (`ok:false` + provider detail), not a transport error. */
export function probeModelProfile(data: Pick<ModelProfile, 'protocol' | 'apibase' | 'model'> & { apikey?: string }): Promise<ModelProbeResult> {
  return fetchJson<ModelProbeResult>('/model-profiles/test', jsonInit('POST', data));
}

export function getMykeyContent(): Promise<string> {
  return getJson<{ content: string }>('/services/mykey').then((r) => r.content || '').catch(() => '');
}

export function saveMykeyContent(content: string): Promise<void> {
  return postJson('/services/mykey', { content }).then(() => undefined);
}

export function tauriInvoke(cmd: string, args: Record<string, unknown> = {}): Promise<unknown> {
  const invoke = (window as any).__TAURI__?.core?.invoke;
  if (!invoke) return Promise.reject(new Error('Tauri IPC not available'));
  return invoke(cmd, args);
}
