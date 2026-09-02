import { BRIDGE_BASE } from './constants';

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

/**
 * The renderer talks to the Python bridge over its REST API only (the websocket is events-only).
 * A non-2xx reply becomes an Error carrying the bridge's `error`/`message` text, or the HTTP status.
 */
export async function bridgeFetch<T = unknown>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BRIDGE_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    let msg = `${res.status} ${res.statusText}`;
    try { const j = JSON.parse(body); msg = j.error || j.message || msg; } catch {}
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body) });

/** Boot-cache defaults used while the bridge is unreachable (mirrors the index.html boot script). */
function cachedConfig(): AppConfig {
  return {
    lang: (localStorage.getItem('ga_lang') as 'zh' | 'en') || 'zh',
    theme: 'light',
    appearance: (localStorage.getItem('ga_appearance') as 'light' | 'dark') || 'light',
    plain: false,
    fontSize: parseInt(localStorage.getItem('ga_font_size') || '14', 10),
    llmNo: 0,
  };
}

export async function getConfig(): Promise<AppConfig> {
  try {
    return (await bridgeFetch<{ config: AppConfig }>('/config')).config;
  } catch {
    return cachedConfig();
  }
}

export async function saveConfig(config: Partial<AppConfig>): Promise<void> {
  try {
    await bridgeFetch('/config', { method: 'POST', ...json({ config }) });
  } catch {
    // Settings are also boot-cached in localStorage; the bridge copy catches up on the next save.
  }
}

export async function getModelProfiles(): Promise<ModelProfile[]> {
  try {
    return (await bridgeFetch<{ profiles: ModelProfile[] }>('/model-profiles')).profiles || [];
  } catch {
    return [];
  }
}

export async function getModelProfileDetail(id: number): Promise<ModelProfile | null> {
  try {
    return (await bridgeFetch<{ profile: ModelProfile }>(`/model-profiles/${id}`)).profile || null;
  } catch {
    return null;
  }
}

type ProfilesReply = { profiles: ModelProfile[] };

export async function addModelProfile(data: Partial<ModelProfile>): Promise<ModelProfile[]> {
  return (await bridgeFetch<ProfilesReply>('/model-profiles', { method: 'POST', ...json(data) })).profiles;
}

export async function editModelProfile(id: number, data: Partial<ModelProfile>): Promise<ModelProfile[]> {
  return (await bridgeFetch<ProfilesReply>(`/model-profiles/${id}`, { method: 'PUT', ...json(data) })).profiles;
}

/** One minimal round-trip against the provider described by `data`; nothing is persisted. */
export async function probeModelProfile(
  data: Pick<ModelProfile, 'protocol' | 'apibase' | 'model'> & { apikey?: string },
): Promise<ModelProbeResult> {
  return bridgeFetch<ModelProbeResult>('/model-profiles/test', { method: 'POST', ...json(data) });
}

export async function deleteModelProfile(id: number): Promise<ModelProfile[]> {
  return (await bridgeFetch<ProfilesReply>(`/model-profiles/${id}`, { method: 'DELETE' })).profiles;
}

export async function addToMixin(id: number): Promise<ModelProfile[]> {
  return (await bridgeFetch<ProfilesReply>(`/model-profiles/${id}/mixin`, { method: 'POST', body: '{}' })).profiles;
}

export async function removeFromMixin(id: number): Promise<ModelProfile[]> {
  return (await bridgeFetch<ProfilesReply>(`/model-profiles/${id}/mixin`, { method: 'DELETE' })).profiles;
}

export async function reorderMixin(members: string[]): Promise<ModelProfile[]> {
  return (await bridgeFetch<ProfilesReply>('/model-profiles/mixin/order', { method: 'PUT', ...json({ members }) })).profiles;
}

export async function getMykeyContent(): Promise<string> {
  try {
    return (await bridgeFetch<{ content: string }>('/services/mykey')).content || '';
  } catch {
    return '';
  }
}

export async function saveMykeyContent(content: string): Promise<void> {
  await bridgeFetch('/services/mykey', { method: 'POST', ...json({ content }) });
}

/** Direct Tauri IPC to the Rust shell; only exists inside the packaged window. */
export async function tauriInvoke(cmd: string, args: Record<string, unknown>): Promise<unknown> {
  const invoke = (window as any).__TAURI__?.core?.invoke as
    | ((cmd: string, args?: Record<string, unknown>) => Promise<unknown>)
    | undefined;
  if (!invoke) throw new Error('Tauri IPC not available');
  return invoke(cmd, args);
}
