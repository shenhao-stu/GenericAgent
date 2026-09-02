// @vitest-environment happy-dom
/**
 * `services/bridge.ts` is the renderer's only path to the bridge REST API: one fetch helper, one error contract.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  addModelProfile,
  bridgeFetch,
  getConfig,
  getModelProfiles,
  probeModelProfile,
  tauriInvoke,
} from '../../services/bridge';

const reply = (status: number, body: unknown, statusText = '') =>
  ({
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
  }) as unknown as Response;

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

  it('targets the bridge base URL with JSON headers and returns the parsed body', async () => {
    fetchMock.mockResolvedValue(reply(200, { profiles: [{ id: 0, name: 'A' }] }));
    const profiles = await getModelProfiles();
    expect(profiles).toEqual([{ id: 0, name: 'A' }]);
    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:14168/model-profiles',
      expect.objectContaining({ headers: { 'Content-Type': 'application/json' } }),
    );
  });

  it("surfaces the bridge's own error text, falling back to the HTTP status", async () => {
    fetchMock.mockResolvedValueOnce(reply(400, { ok: false, error: 'model is required' }));
    await expect(addModelProfile({ name: 'x' })).rejects.toThrow('model is required');

    fetchMock.mockResolvedValueOnce(reply(502, '<html>bad gateway</html>', 'Bad Gateway'));
    await expect(bridgeFetch('/anything')).rejects.toThrow('502 Bad Gateway');
  });

  it('posts probe payloads verbatim and hands back the bridge verdict', async () => {
    fetchMock.mockResolvedValue(reply(200, { ok: false, status: 401, error: 'HTTP 401: invalid key' }));
    const verdict = await probeModelProfile({ protocol: 'oai', apibase: 'https://x', model: 'm', apikey: 'k' });
    expect(verdict).toEqual({ ok: false, status: 401, error: 'HTTP 401: invalid key' });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ protocol: 'oai', apibase: 'https://x', model: 'm', apikey: 'k' });
  });

  it('falls back to the boot cache for config while the bridge is unreachable', async () => {
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
