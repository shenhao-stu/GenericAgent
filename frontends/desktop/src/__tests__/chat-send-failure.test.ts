// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  createSession: vi.fn(),
  sendPrompt: vi.fn(),
  pollMessages: vi.fn(() => new Promise(() => {})),
  listSessions: vi.fn(() => Promise.resolve([])),
  setSessionModel: vi.fn(),
}));

vi.mock('../services/chat', () => ({
  createSession: mocks.createSession,
  sendPrompt: mocks.sendPrompt,
  pollMessages: mocks.pollMessages,
  cancelGeneration: vi.fn(),
  listSessions: mocks.listSessions,
  deleteSession: vi.fn(),
  renameSession: vi.fn(),
  pinSession: vi.fn(),
  setSessionModel: mocks.setSessionModel,
}));

vi.mock('../services/ws', () => ({
  subscribe: () => () => {},
  onBridgeStatusChange: vi.fn(),
  getBridgeStatus: () => 'offline',
}));

vi.mock('../stores/settings', () => ({
  useSettingsStore: { getState: () => ({ setLiveModel: vi.fn(), lang: 'en' }) },
}));

import { __resetChatStoreForTests, useChatStore } from '../stores/chat';
import { NEW_SESSION_VIEW_ID, useThreadViewStore } from '../stores/thread-view';
import { useNotificationStore } from '../stores/notifications';
import { en } from '../i18n/en';

describe('send failures stay visible and never lose input', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mocks.createSession.mockReset();
    mocks.sendPrompt.mockReset();
    __resetChatStoreForTests();
    useThreadViewStore.setState({ viewBySessionId: {} });
    useNotificationStore.setState({ items: [] });
  });

  afterEach(() => {
    __resetChatStoreForTests();
    vi.useRealTimers();
  });

  it('marks the prompt failed and explains why when the bridge rejects it', async () => {
    mocks.sendPrompt.mockRejectedValue(new Error('session is already running'));
    useChatStore.getState().setActiveSession('A');

    await useChatStore.getState().sendMessage('hello');

    const { messages, status, runningSessions } = useChatStore.getState();
    expect(status).toBe('idle');
    expect(runningSessions.has('A')).toBe(false);
    expect(messages.map((m) => [m.role, m.status])).toEqual([['user', 'failed'], ['error', 'failed']]);
    expect(messages[0].content).toBe('hello');
    expect(messages[1].content).toBe('session is already running');

    const pollsSoFar = mocks.pollMessages.mock.calls.length;
    await vi.advanceTimersByTimeAsync(3000);
    expect(mocks.pollMessages.mock.calls.length).toBe(pollsSoFar);
  });

  it('hands the draft back and notifies when no session could be created', async () => {
    mocks.createSession.mockRejectedValue(new Error('HTTP 500'));

    await useChatStore.getState().sendMessage('keep me');

    expect(useChatStore.getState().activeSessionId).toBeNull();
    expect(useChatStore.getState().sessionsById).toEqual({});
    expect(mocks.sendPrompt).not.toHaveBeenCalled();
    expect(useThreadViewStore.getState().viewBySessionId[NEW_SESSION_VIEW_ID]?.composerDraft).toBe('keep me');

    const [notice] = useNotificationStore.getState().items;
    expect(notice.kind).toBe('error');
    expect(notice.message).toBe(en['err.newSession']);
    expect(notice.detail).toBe('HTTP 500');
  });

  it('keeps the happy path unchanged', async () => {
    mocks.createSession.mockResolvedValue('S1');
    mocks.sendPrompt.mockResolvedValue('m1');

    await useChatStore.getState().sendMessage('hi');

    const state = useChatStore.getState();
    expect(state.activeSessionId).toBe('S1');
    expect(state.status).toBe('running');
    expect(state.messages.map((m) => [m.role, m.status])).toEqual([['user', 'completed']]);
    expect(useNotificationStore.getState().items).toEqual([]);
  });
});
