// @vitest-environment happy-dom
/**
 * A finished turn the user is not looking at never goes unnoticed:
 * - another session finishing -> unread mark + in-app notice that opens it
 * - the window being unattended -> unread mark + platform attention request; coming back clears it
 * - opening a running session follows its live tail (#683)
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PollResult } from '../services/chat';

const mocks = vi.hoisted(() => ({
  pollMessages: vi.fn(),
  listSessions: vi.fn(() => Promise.resolve([])),
  wsHandlers: new Map<string, (payload: unknown) => void>(),
  attended: true,
  requestUserAttention: vi.fn(),
  attendedHandlers: new Set<() => void>(),
}));

vi.mock('../services/chat', () => ({
  createSession: vi.fn(),
  sendPrompt: vi.fn(),
  pollMessages: mocks.pollMessages,
  cancelGeneration: vi.fn(),
  listSessions: mocks.listSessions,
  deleteSession: vi.fn(() => Promise.resolve()),
  renameSession: vi.fn(),
  pinSession: vi.fn(),
  setSessionModel: vi.fn(),
}));

vi.mock('../services/ws', () => ({
  subscribe: (type: string, handler: (payload: unknown) => void) => {
    mocks.wsHandlers.set(type, handler);
    return () => mocks.wsHandlers.delete(type);
  },
  onBridgeStatusChange: vi.fn(),
  getBridgeStatus: () => 'ready',
}));

vi.mock('../services/attention', () => ({
  windowIsAttended: () => mocks.attended,
  requestUserAttention: mocks.requestUserAttention,
  onWindowAttended: (handler: () => void) => {
    mocks.attendedHandlers.add(handler);
    return () => mocks.attendedHandlers.delete(handler);
  },
}));

vi.mock('../stores/settings', () => ({
  useSettingsStore: { getState: () => ({ setLiveModel: vi.fn(), lang: 'en' }) },
}));

import { __resetChatStoreForTests, useChatStore } from '../stores/chat';
import { useNotificationStore } from '../stores/notifications';
import { useAppStore } from '../stores/app';
import { useThreadViewStore } from '../stores/thread-view';

const idle = (): PollResult => ({ messages: [], status: 'idle' });
const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
const sessionState = (sessionId: string, status: 'running' | 'idle') =>
  mocks.wsHandlers.get('session-state')?.({ sessionId, status });

describe('turn-completion attention', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('requestAnimationFrame', () => 1);
    vi.stubGlobal('cancelAnimationFrame', () => {});
    mocks.pollMessages.mockReset();
    mocks.pollMessages.mockResolvedValue(idle());
    mocks.requestUserAttention.mockReset();
    mocks.attended = true;
    __resetChatStoreForTests();
    useNotificationStore.getState().clear();
    useChatStore.setState({ sessions: [{ id: 'B', title: 'Weekly report', untitled: false }] });
  });

  afterEach(() => {
    __resetChatStoreForTests();
    useNotificationStore.getState().clear();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('marks another session unread when its turn ends and offers to open it', async () => {
    useChatStore.getState().setActiveSession('A');
    await flush();
    sessionState('B', 'running');
    sessionState('B', 'idle');

    expect(useChatStore.getState().unreadSessions.has('B')).toBe(true);
    expect(mocks.requestUserAttention).not.toHaveBeenCalled();
    const notice = useNotificationStore.getState().items.at(-1)!;
    expect(notice.message).toBe('“Weekly report” has replied');

    useAppStore.getState().setPage('token');
    notice.action!.onClick();
    expect(useChatStore.getState().activeSessionId).toBe('B');
    expect(useAppStore.getState().activePage).toBe('chat');
    expect(useChatStore.getState().unreadSessions.has('B')).toBe(false);
  });

  it('asks for the platform attention when the active session finishes in an unattended window, and clears on return', async () => {
    useChatStore.getState().setActiveSession('A');
    await flush();
    mocks.attended = false;
    sessionState('A', 'running');
    sessionState('A', 'idle');

    expect(useChatStore.getState().unreadSessions.has('A')).toBe(true);
    expect(mocks.requestUserAttention).toHaveBeenCalledTimes(1);
    expect(useNotificationStore.getState().items).toHaveLength(0);

    mocks.attended = true;
    for (const handler of mocks.attendedHandlers) handler();
    expect(useChatStore.getState().unreadSessions.size).toBe(0);
  });

  it('does nothing for the session the user is watching', async () => {
    useChatStore.getState().setActiveSession('A');
    await flush();
    sessionState('A', 'running');
    sessionState('A', 'idle');

    expect(useChatStore.getState().unreadSessions.size).toBe(0);
    expect(mocks.requestUserAttention).not.toHaveBeenCalled();
    expect(useNotificationStore.getState().items).toHaveLength(0);
  });

  it('forgets unread marks when the session is deleted', async () => {
    useChatStore.getState().setActiveSession('A');
    await flush();
    sessionState('B', 'running');
    sessionState('B', 'idle');
    await useChatStore.getState().deleteSession('B');
    expect(useChatStore.getState().unreadSessions.has('B')).toBe(false);
  });

  it('opens a running session at its live tail even if the user had scrolled up earlier (#683)', async () => {
    useThreadViewStore.getState().setScrollState('B', { scrollTop: 120 }, false);
    sessionState('B', 'running');

    useChatStore.getState().setActiveSession('B');
    const view = useThreadViewStore.getState().viewBySessionId.B;
    expect(view.followingTail).toBe(true);
    expect(view.scrollAnchor).toBeNull();
  });
});
