import { fetchJson, getJson, jsonInit, postJson } from './http';

export interface ServiceInfo {
  id: string;
  name: string;
  status: 'running' | 'offline' | 'error' | 'warning';
  running: boolean;
  pid: number | null;
  memMb: number | null;
  cpuPct: number | null;
  managed: boolean;
  lastError: string | null;
  errorKey?: string;
  lastWarning?: string;
  warningKey?: string;
}

export type ModelFallbackReason =
  | 'ui_default'
  | 'invalid_configured'
  | 'configured_unavailable'
  | 'first_available'
  | 'no_models'
  | null;

export interface ConductorModelState {
  configured: number | null;
  effective: number | null;
  fallbackReason: ModelFallbackReason;
}

type ServiceResult = { ok: boolean; service?: ServiceInfo };

export const fetchConductorModel = () => getJson<{ model: ConductorModelState }>('/services/conductor/model').then((d) => d.model);
export const saveConductorModel = (llmNo: number) => postJson<{ model: ConductorModelState }>('/services/conductor/model', { llmNo }).then((d) => d.model);
export const fetchServicesPanel = () => getJson<{ services?: ServiceInfo[] }>('/services/panel').then((d) => d.services ?? []);

/** Start/stop answer with `ok:false` plus the service's error state; that state is data, not a transport failure. */
const serviceCommand = (path: string, id: string) =>
  fetchJson<ServiceResult>(path, jsonInit('POST', { id })).then((d) => ({ ok: d.ok ?? true, service: d.service }));
export const startServiceById = (id: string) => serviceCommand('/services/start', id);
export const stopServiceById = (id: string) => serviceCommand('/services/stop', id);

/** Stop / start every bridge-owned extra (conductor, scheduler); IM channels are untouched. */
export const stopAllExtras = () => postJson('/services/stop-extras').then(() => undefined);
export const startAllExtras = () => postJson('/services/start-extras').then(() => undefined);

export const fetchServiceLogs = (id: string, tail = 200) =>
  getJson<{ lines?: string[] }>(`/services/logs?id=${encodeURIComponent(id)}&tail=${tail}`).then((d) => d.lines ?? []);
export const fetchMykeyContent = () => getJson<{ content?: string }>('/services/mykey').then((d) => d.content ?? '');
export const saveMykeyContent = (content: string) => postJson('/services/mykey', { content }).then(() => true);
export const exitBridge = () => postJson('/services/bridge/exit').then(() => true);
