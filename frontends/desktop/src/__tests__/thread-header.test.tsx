// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { SessionInfo } from '../services/chat';

const mocks = vi.hoisted(() => ({
  renameSession: vi.fn(),
  tauriInvoke: vi.fn(() => Promise.resolve(null)),
  state: {
    activeSessionId: null as string | null,
    sessions: [] as SessionInfo[],
    status: 'idle' as 'idle' | 'running',
    turnStartedAt: null as number | null,
  },
}));

vi.mock('../stores/chat', () => ({
  useChatStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({ ...mocks.state, renameSession: mocks.renameSession }),
}));
vi.mock('../stores/settings', () => ({
  useSettingsStore: (selector: (state: { lang: string }) => unknown) => selector({ lang: 'en' }),
}));
vi.mock('../services/bridge', () => ({ tauriInvoke: mocks.tauriInvoke }));

import { ThreadHeader } from '../components/chat/Thread/ThreadHeader';

const session = (extra: Partial<SessionInfo> = {}): SessionInfo => ({
  id: 's1', title: 'Weekly report', untitled: false, ...extra,
});

describe('ThreadHeader', () => {
  beforeEach(() => {
    mocks.renameSession.mockReset();
    mocks.tauriInvoke.mockClear();
    mocks.state.activeSessionId = 's1';
    mocks.state.sessions = [session()];
    mocks.state.status = 'idle';
    mocks.state.turnStartedAt = null;
  });

  afterEach(() => cleanup());

  it('renders nothing without an active session', () => {
    mocks.state.activeSessionId = null;
    const { container } = render(<ThreadHeader />);
    expect(container.firstChild).toBeNull();
  });

  it('shows the session title, falling back to the localized placeholder until the list has the session', () => {
    render(<ThreadHeader />);
    expect(screen.getByText('Weekly report')).not.toBeNull();
    cleanup();

    mocks.state.sessions = [];
    render(<ThreadHeader />);
    expect(screen.getByText('New session')).not.toBeNull();
  });

  it('renames inline: click the title, type, Enter; unchanged or empty titles are not sent', () => {
    render(<ThreadHeader />);
    fireEvent.click(screen.getByText('Weekly report'));
    const input = screen.getByRole('textbox') as HTMLInputElement;
    expect(input.value).toBe('Weekly report');

    fireEvent.change(input, { target: { value: '  Q3 report ' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(mocks.renameSession).toHaveBeenCalledWith('s1', 'Q3 report');
    expect(screen.queryByRole('textbox')).toBeNull();

    fireEvent.click(screen.getByText('Weekly report'));
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '   ' } });
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    expect(mocks.renameSession).toHaveBeenCalledTimes(1);
  });

  it('cancels with Escape even though the field blurs on unmount', () => {
    render(<ThreadHeader />);
    fireEvent.click(screen.getByText('Weekly report'));
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'discarded' } });
    fireEvent.keyDown(input, { key: 'Escape' });
    fireEvent.blur(input);
    expect(mocks.renameSession).not.toHaveBeenCalled();
    expect(screen.queryByRole('textbox')).toBeNull();
  });

  it('shows the bound folder by name and reveals it on click', () => {
    mocks.state.sessions = [session({ workDir: 'D:\\projects\\demo' })];
    render(<ThreadHeader />);
    fireEvent.click(screen.getByText('demo'));
    expect(mocks.tauriInvoke).toHaveBeenCalledWith('reveal_in_file_manager', { path: 'D:\\projects\\demo' });
  });

  it('shows the live turn duration only while the session is running', () => {
    const { container } = render(<ThreadHeader />);
    expect(container.querySelector('[data-slot="thread-header-live"]')).toBeNull();
    cleanup();

    mocks.state.status = 'running';
    mocks.state.turnStartedAt = Date.now() - 65_000;
    render(<ThreadHeader />);
    expect(screen.getByRole('status').textContent).toMatch(/1m \d+s/);
  });
});
