// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BridgeError, fetchJson, getJson, postJson, requestJson } from '../../services/http';

const reply = (status: number, body: unknown) => ({ ok: status >= 200 && status < 300, status, json: () => Promise.resolve(body) });

describe('bridge JSON transport contract', () => {
  const mockFetch = vi.fn();
  beforeEach(() => vi.stubGlobal('fetch', mockFetch));
  afterEach(() => { vi.unstubAllGlobals(); mockFetch.mockReset(); });

  it('GET sends a bare fetch(url) and returns the parsed body', async () => {
    mockFetch.mockResolvedValue(reply(200, { sessions: [] }));
    await expect(getJson('/sessions')).resolves.toEqual({ sessions: [] });
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:14168/sessions');
  });

  it('POST serialises the body with a JSON content type', async () => {
    mockFetch.mockResolvedValue(reply(200, { ok: true, sessionId: 's1' }));
    await postJson('/session/new', { cwd: '' });
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:14168/session/new', expect.objectContaining({
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cwd: '' }),
    }));
  });

  it('non-2xx throws a BridgeError carrying the bridge message, status and payload', async () => {
    mockFetch.mockResolvedValue(reply(409, { ok: false, error: 'busy', code: 'maintenance_conflict', runningExtras: ['x'] }));
    const error = await requestJson('/memory/export').catch((e) => e);
    expect(error).toBeInstanceOf(BridgeError);
    expect(error).toMatchObject({ message: 'busy', status: 409, code: 'maintenance_conflict' });
    expect(error.payload.runningExtras).toEqual(['x']);
  });

  it('falls back to "HTTP <status>" when the body has no error text', async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 500, json: () => Promise.reject(new SyntaxError('html')) });
    await expect(getJson('/status')).rejects.toThrow('HTTP 500');
  });

  it('requestJson rejects a 200 {ok:false}; fetchJson hands it back as a result', async () => {
    mockFetch.mockResolvedValue(reply(200, { ok: false, error: 'HTTP 401: invalid key', status: 401 }));
    await expect(requestJson('/model-profiles/test')).rejects.toThrow('HTTP 401: invalid key');
    await expect(fetchJson('/model-profiles/test')).resolves.toMatchObject({ ok: false, status: 401 });
  });

  it('network failures propagate untouched', async () => {
    mockFetch.mockRejectedValue(new Error('ECONNREFUSED'));
    await expect(getJson('/status')).rejects.toThrow('ECONNREFUSED');
  });
});
