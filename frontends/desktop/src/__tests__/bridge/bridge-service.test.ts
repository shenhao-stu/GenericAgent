// @vitest-environment happy-dom
/**
 * `services/bridge.ts` sits on the shared JSON transport (`services/http.ts`, contract-tested in
 * http-client.test.ts). What is specific here: offline fallbacks, the probe verdict, and the Tauri IPC seam.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  addModelProfile,
  getConfig,
  getModelProfiles,
  probeModelProfile,
  tauriInvoke,
} from '../../services/bridge';

const reply = (status: number, body: unknown) =>
  ({ ok: status >= 200 && status < 300, status, json: () => Promise.resolve(body) }) as unknown as Response;

describe('bridge service', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
    localStorage.clear();
    delete (window as any).__TAURI__;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('unwraps profile lists and surfaces the bridge error text on rejection', async () => {
    fetchMock.mockResolvedValueOnce(reply(200, { profiles: [{ id: 0, name: 'A' }] }));
    expect(await getModelProfiles()).toEqual([{ id: 0, name: 'A' }]);

    fetchMock.mockResolvedValueOnce(reply(400, { ok: false, error: 'model is required' }));
    await expect(addModelProfile({ name: 'x' })).rejects.toThrow('model is required');
  });

  it('posts probe payloads verbatim and hands back the verdict even when it is a failure', async () => {
    fetchMock.mockResolvedValue(reply(200, { ok: false, status: 401, error: 'HTTP 401: invalid key' }));
    const verdict = await probeModelProfile({ protocol: 'oai', apibase: 'https://x', model: 'm', apikey: 'k' });
    expect(verdict).toEqual({ ok: false, status: 401, error: 'HTTP 401: invalid key' });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ protocol: 'oai', apibase: 'https://x', model: 'm', apikey: 'k' });
  });

  it('falls back to the boot cache for config and to an empty list for profiles while the bridge is unreachable', async () => {
    fetchMock.mockRejectedValue(new Error('ECONNREFUSED'));
    localStorage.setItem('ga_lang', 'en');
    localStorage.setItem('ga_font_size', '16');
    expect(await getConfig()).toMatchObject({ lang: 'en', fontSize: 16, appearance: 'light', llmNo: 0 });
    expect(await getModelProfiles()).toEqual([]);
  });

  it('routes tauriInvoke to the shell IPC and fails loudly outside the packaged window', async () => {
    await expect(tauriInvoke('pick_directory', {})).rejects.toThrow('Tauri IPC not available');

    const invoke = vi.fn(() => Promise.resolve('D:\\work'));
    (window as any).__TAURI__ = { core: { invoke } };
    expect(await tauriInvoke('pick_directory', { title: 't' })).toBe('D:\\work');
    expect(invoke).toHaveBeenCalledWith('pick_directory', { title: 't' });
  });
});
