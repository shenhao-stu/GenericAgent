// @vitest-environment happy-dom
/**
 * Live-update transport rules of the chat store:
 * - the websocket is the streaming channel; polling is a 1 s fallback while it is down and a 5 s safety net otherwise
 * - polls are incremental (`after=<newest bridge id>`) once a session holds bridge messages
 * - a poll's partial never regresses a websocket partial while the socket is live
 * - a (re)connected socket resyncs every running session and the active one
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { BridgeStatus } from '../services/ws';
import type { Message, PollResult } from '../services/chat';

const mocks = vi.hoisted(() => ({
  pollMessages: vi.fn(),
  sendPrompt: vi.fn(),
  listSessions: vi.fn(() => Promise.resolve([])),
  wsHandlers: new Map<string, (payload: unknown) => void>(),
  statusHandlers: new Set<(status: string) => void>(),
  bridgeStatus: 'offline' as string,
}));

vi.mock('../services/chat', () => ({
  createSession: vi.fn(),
  sendPrompt: mocks.sendPrompt,
  pollMessages: mocks.pollMessages,
  cancelGeneration: vi.fn(),
  listSessions: mocks.listSessions,
  deleteSession: vi.fn(),
  renameSession: vi.fn(),
  pinSession: vi.fn(),
  setSessionModel: vi.fn(),
}));

vi.mock('../services/ws', () => ({
  subscribe: (type: string, handler: (payload: unknown) => void) => {
    mocks.wsHandlers.set(type, handler);
    return () => mocks.wsHandlers.delete(type);
  },
  onBridgeStatusChange: (handler: (status: string) => void) => {
    mocks.statusHandlers.add(handler);
    return () => mocks.statusHandlers.delete(handler);
  },
  getBridgeStatus: () => mocks.bridgeStatus,
}));

vi.mock('../stores/settings', () => ({
  useSettingsStore: { getState: () => ({ setLiveModel: vi.fn(), lang: 'en' }) },
}));

import {
  __resetChatStoreForTests,
  lastServerMessageId,
  POLL_FALLBACK_MS,
  POLL_HEARTBEAT_MS,
  useChatStore,
} from '../stores/chat';

const serverMessage = (id: number, content = `m${id}`): Message => ({
  id: String(id), role: 'assistant', content, status: 'completed', createdAt: id,
});

const partialOf = (content: string): Message => ({
  id: '__partial__', role: 'assistant', content, status: 'in_progress',
});

const running = (messages: Message[], partial?: Message): PollResult => ({ messages, status: 'running', partial });
const idle = (messages: Message[]): PollResult => ({ messages, status: 'idle' });

const setBridge = (status: BridgeStatus) => {
  mocks.bridgeStatus = status;
  for (const handler of mocks.statusHandlers) handler(status);
};

const flush = async () => { await Promise.resolve(); await Promise.resolve(); };

describe('lastServerMessageId', () => {
  it('is the newest integer bridge id, ignoring optimistic and partial rows', () => {
    expect(lastServerMessageId([])).toBeUndefined();
    expect(lastServerMessageId([
      { id: 'local-s-1', role: 'user', content: 'x', status: 'completed' },
      { id: '__partial__:s', role: 'assistant', content: '…', status: 'in_progress' },
    ])).toBeUndefined();
    expect(lastServerMessageId([serverMessage(3), serverMessage(12), serverMessage(7), partialOf('…')])).toBe('12');
  });
});

describe('chat transport', () => {
  const frames = new Map<number, FrameRequestCallback>();
  const runFrames = () => {
    for (const [id, callback] of [...frames]) { frames.delete(id); callback(16); }
  };

  beforeEach(() => {
    vi.useFakeTimers();
    let nextFrame = 0;
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { frames.set(++nextFrame, callback); return nextFrame; });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id); });
    mocks.pollMessages.mockReset();
    mocks.listSessions.mockReset();
    mocks.listSessions.mockResolvedValue([]);
    mocks.bridgeStatus = 'offline';
    __resetChatStoreForTests();
  });

  afterEach(() => {
    __resetChatStoreForTests();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('polls every second only while the websocket is down, every five seconds while it is live', async () => {
    mocks.pollMessages.mockResolvedValue(running([]));
    mocks.wsHandlers.get('session-state')?.({ sessionId: 'A', status: 'running' });

    await vi.advanceTimersByTimeAsync(POLL_FALLBACK_MS);
    expect(mocks.pollMessages).toHaveBeenCalledTimes(1);

    mocks.bridgeStatus = 'ready';
    await vi.advanceTimersByTimeAsync(POLL_HEARTBEAT_MS - POLL_FALLBACK_MS);
    expect(mocks.pollMessages).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(POLL_FALLBACK_MS);
    expect(mocks.pollMessages).toHaveBeenCalledTimes(2);

    mocks.bridgeStatus = 'offline';
    await vi.advanceTimersByTimeAsync(POLL_FALLBACK_MS);
    expect(mocks.pollMessages).toHaveBeenCalledTimes(3);
    await vi.advanceTimersByTimeAsync(POLL_FALLBACK_MS);
    expect(mocks.pollMessages).toHaveBeenCalledTimes(4);
  });

  it('loads a session in full first, then only asks for messages newer than the last bridge id', async () => {
    mocks.pollMessages
      .mockResolvedValueOnce(running([serverMessage(1), serverMessage(2)]))
      .mockResolvedValue(running([serverMessage(3)]));

    useChatStore.getState().setActiveSession('A');
    await flush();
    expect(mocks.pollMessages).toHaveBeenNthCalledWith(1, 'A', undefined);

    await vi.advanceTimersByTimeAsync(POLL_FALLBACK_MS);
    expect(mocks.pollMessages).toHaveBeenNthCalledWith(2, 'A', '2');

    await vi.advanceTimersByTimeAsync(POLL_FALLBACK_MS);
    expect(mocks.pollMessages).toHaveBeenNthCalledWith(3, 'A', '3');
    expect(useChatStore.getState().messages.map((m) => m.id)).toEqual(['1', '2', '3']);
  });

  it('never lets a poll partial regress the websocket partial while the socket is live', async () => {
    mocks.bridgeStatus = 'ready';
    mocks.pollMessages.mockResolvedValue(running([], partialOf('hel')));
    useChatStore.getState().setActiveSession('A');
    await flush();
    // No websocket partial yet: the poll seeds the streaming row.
    expect(useChatStore.getState().messages.at(-1)?.content).toBe('hel');

    mocks.wsHandlers.get('partial-update')?.({ sessionId: 'A', content: 'hello world' });
    runFrames();
    expect(useChatStore.getState().messages.at(-1)?.content).toBe('hello world');

    await vi.advanceTimersByTimeAsync(POLL_HEARTBEAT_MS);
    expect(mocks.pollMessages).toHaveBeenCalledTimes(2);
    expect(useChatStore.getState().messages.at(-1)?.content).toBe('hello world');

    // Socket down: the poll is the only source and its partial is used.
    mocks.bridgeStatus = 'offline';
    mocks.pollMessages.mockResolvedValue(running([], partialOf('hello world, again')));
    await vi.advanceTimersByTimeAsync(POLL_FALLBACK_MS);
    expect(useChatStore.getState().messages.at(-1)?.content).toBe('hello world, again');

    // A finished turn clears the streaming row regardless of source.
    mocks.pollMessages.mockResolvedValue(idle([serverMessage(1, 'final')]));
    await vi.advanceTimersByTimeAsync(POLL_FALLBACK_MS);
    const state = useChatStore.getState();
    expect(state.status).toBe('idle');
    expect(state.messages.map((m) => m.content)).toEqual(['final']);
  });

  it('marks a selected session hydrated only once its first fetch lands (no welcome-screen flash)', async () => {
    let resolveLoad!: (value: PollResult) => void;
    mocks.pollMessages.mockImplementation(() => new Promise<PollResult>((resolve) => { resolveLoad = resolve; }));

    expect(useChatStore.getState().hydrated).toBe(true);
    useChatStore.getState().setActiveSession('A');
    expect(useChatStore.getState().hydrated).toBe(false);

    resolveLoad(idle([]));
    await flush();
    expect(useChatStore.getState().hydrated).toBe(true);
    expect(useChatStore.getState().messages).toEqual([]);

    useChatStore.getState().setActiveSession(null);
    expect(useChatStore.getState().hydrated).toBe(true);
  });

  it('resyncs running sessions and the active one when the websocket comes back', async () => {
    mocks.pollMessages.mockResolvedValue(running([]));
    mocks.wsHandlers.get('session-state')?.({ sessionId: 'A', status: 'running' });
    mocks.wsHandlers.get('session-state')?.({ sessionId: 'B', status: 'running' });
    mocks.pollMessages.mockResolvedValue(idle([]));
    useChatStore.getState().setActiveSession('C');
    await flush();
    mocks.pollMessages.mockClear();
    mocks.listSessions.mockClear();

    setBridge('ready');
    await flush();

    const polled = mocks.pollMessages.mock.calls.map(([id]) => id).sort();
    expect(polled).toEqual(['A', 'B', 'C']);
    expect(mocks.listSessions).toHaveBeenCalledTimes(1);

    setBridge('connecting');
    await flush();
    expect(mocks.pollMessages).toHaveBeenCalledTimes(3);
  });
});
