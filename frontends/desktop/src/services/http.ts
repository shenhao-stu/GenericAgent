import { BRIDGE_BASE } from './constants';

/**
 * The one JSON transport for the local bridge. Contract: a non-2xx status or an `{ok:false}` body is a
 * failure; the bridge's `error` text becomes the message and the parsed body stays on `payload` so
 * callers can read structured detail (`code`, `runningSessions`, …) without re-parsing.
 */
export class BridgeError extends Error {
  readonly status: number;
  readonly payload: Record<string, unknown>;

  constructor(message: string, status: number, payload: Record<string, unknown>) {
    super(message);
    this.name = 'BridgeError';
    this.status = status;
    this.payload = payload;
  }

  get code(): string | null {
    return typeof this.payload.code === 'string' ? this.payload.code : null;
  }
}

const JSON_HEADERS = { 'Content-Type': 'application/json' };

export const jsonInit = (method: string, body: unknown): RequestInit =>
  ({ method, headers: JSON_HEADERS, body: JSON.stringify(body) });

/** HTTP-level contract only: non-2xx throws. Use for endpoints whose 200 body is itself a result with `ok`. */
export async function fetchJson<T = Record<string, any>>(
  path: string,
  init?: RequestInit,
  base: string = BRIDGE_BASE,
): Promise<T> {
  const url = `${base}${path}`;
  const res = init ? await fetch(url, init) : await fetch(url);
  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  if (!res.ok) throw new BridgeError(bridgeMessage(data, res.status), res.status, data);
  return data as T;
}

/** Full bridge contract: non-2xx or `{ok:false}` throws. The default for every mutation and lookup. */
export async function requestJson<T = Record<string, any>>(
  path: string,
  init?: RequestInit,
  base: string = BRIDGE_BASE,
): Promise<T> {
  const data = await fetchJson<Record<string, unknown>>(path, init, base);
  if (data.ok === false) throw new BridgeError(bridgeMessage(data, 200), 200, data);
  return data as T;
}

function bridgeMessage(data: Record<string, unknown>, status: number): string {
  return typeof data.error === 'string' && data.error ? data.error : `HTTP ${status}`;
}

export const getJson = <T = Record<string, any>>(path: string, base?: string) => requestJson<T>(path, undefined, base);

export const postJson = <T = Record<string, any>>(path: string, body: unknown = {}, base?: string) =>
  requestJson<T>(path, jsonInit('POST', body), base);

export const putJson = <T = Record<string, any>>(path: string, body: unknown) => requestJson<T>(path, jsonInit('PUT', body));

export const patchJson = <T = Record<string, any>>(path: string, body: unknown) => requestJson<T>(path, jsonInit('PATCH', body));

export const deleteJson = <T = Record<string, any>>(path: string) => requestJson<T>(path, { method: 'DELETE' });

export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
