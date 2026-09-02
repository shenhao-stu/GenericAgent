// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  createSession: vi.fn(() => Promise.resolve('sess-1')),
  sendPrompt: vi.fn(() => Promise.resolve('msg-1')),
  pollMessages: vi.fn(() => Promise.resolve({ messages: [], status: 'idle' })),
  cancelGeneration: vi.fn(),
  listSessions: vi.fn(() => Promise.resolve([])),
  deleteSession: vi.fn(),
  renameSession: vi.fn(),
  pinSession: vi.fn(),
  setSessionModel: vi.fn(),
  wsHandlers: new Map<string, (payload: unknown) => void>(),
}));

vi.mock('../services/chat', () => ({
  createSession: mocks.createSession,
  sendPrompt: mocks.sendPrompt,
  pollMessages: mocks.pollMessages,
  cancelGeneration: mocks.cancelGeneration,
  listSessions: mocks.listSessions,
  deleteSession: mocks.deleteSession,
  renameSession: mocks.renameSession,
  pinSession: mocks.pinSession,
  setSessionModel: mocks.setSessionModel,
}));

vi.mock('../services/ws', () => ({
  subscribe: (type: string, handler: (payload: unknown) => void) => {
    mocks.wsHandlers.set(type, handler);
    return () => mocks.wsHandlers.delete(type);
  },
  onBridgeStatusChange: vi.fn(),
}));

vi.mock('../stores/settings', () => ({
  useSettingsStore: { getState: () => ({ setLiveModel: vi.fn() }) },
}));

import { __resetChatStoreForTests, useChatStore } from '../stores/chat';
import { folderName } from '../components/chat/Composer/workdir';

describe('session working directory (#780)', () => {
  beforeEach(() => {
    __resetChatStoreForTests();
    mocks.createSession.mockClear();
  });

  it('binds the pending folder to the session created by the first message, then clears it', async () => {
    useChatStore.getState().setPendingWorkDir('D:\\projects\\demo');
    await useChatStore.getState().sendMessage('hello');

    expect(mocks.createSession).toHaveBeenCalledWith('D:\\projects\\demo');
    expect(useChatStore.getState().pendingWorkDir).toBeNull();
    expect(useChatStore.getState().activeSessionId).toBe('sess-1');
  });

  it('creates default sessions with an empty folder when nothing was picked', async () => {
    await useChatStore.getState().sendMessage('hello');
    expect(mocks.createSession).toHaveBeenCalledWith('');
  });

  it('drops a pending folder when the user starts a fresh new session', async () => {
    useChatStore.getState().setPendingWorkDir('/tmp/x');
    await useChatStore.getState().newSession();
    expect(useChatStore.getState().pendingWorkDir).toBeNull();
  });

  it('shows the last folder segment for either path style', () => {
    expect(folderName('D:\\projects\\demo\\')).toBe('demo');
    expect(folderName('/home/me/work')).toBe('work');
    expect(folderName('C:\\')).toBe('C:');
  });
});
