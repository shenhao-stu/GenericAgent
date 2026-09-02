// @vitest-environment happy-dom
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  modalConfirm: vi.fn(),
  toastSuccess: vi.fn(),
  toastError: vi.fn(),
  toastWarning: vi.fn(),
  tauriInvoke: vi.fn(),
  supportsDataBackupApi: vi.fn(),
  inspectDataImport: vi.fn(),
  importData: vi.fn(),
  exportData: vi.fn(),
  loadSessions: vi.fn(),
  fetchServices: vi.fn(),
  stopAllExtras: vi.fn(),
  startAllExtras: vi.fn(),
  chatState: {
    runningSessions: new Set<string>(),
    status: 'idle',
    sessions: [] as Array<{ id: string; title: string; untitled: boolean; status?: string }>,
  },
  servicesState: {
    services: [] as Array<{ id: string; managed: boolean; running: boolean }>,
  },
}));

vi.mock('@douyinfe/semi-ui', () => {
  const Modal = Object.assign(
    ({ visible, title, children, footer }: any) => visible
      ? <div role="dialog"><h2>{title}</h2>{children}{footer}</div>
      : null,
    { confirm: mocks.modalConfirm },
  );
  return {
    Button: ({ children, type: _type, ...props }: any) => <button {...props}>{children}</button>,
    Modal,
    Toast: {
      success: mocks.toastSuccess,
      error: mocks.toastError,
      warning: mocks.toastWarning,
    },
    Tooltip: ({ children }: any) => children,
  };
});

vi.mock('../utils/tauri', () => ({ isTauri: () => true }));
vi.mock('../services/bridge', () => ({
  tauriInvoke: mocks.tauriInvoke,
  saveMykeyContent: vi.fn(),
  getMykeyContent: vi.fn(),
}));
vi.mock('../services/dataBackup', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/dataBackup')>();
  return {
    ...actual,
    supportsDataBackupApi: mocks.supportsDataBackupApi,
    inspectDataImport: mocks.inspectDataImport,
    importData: mocks.importData,
    exportData: mocks.exportData,
  };
});
vi.mock('../stores/chat', () => {
  const useChatStore = (selector: (state: typeof mocks.chatState) => unknown) => (
    selector(mocks.chatState)
  );
  useChatStore.getState = () => ({ loadSessions: mocks.loadSessions });
  return { useChatStore };
});
vi.mock('../stores/services', () => {
  const currentState = () => ({
    ...mocks.servicesState, fetchServices: mocks.fetchServices, stopAllExtras: mocks.stopAllExtras, startAllExtras: mocks.startAllExtras,
  });
  const useServicesStore = (selector: (value: ReturnType<typeof currentState>) => unknown) => (
    selector(currentState())
  );
  useServicesStore.getState = currentState;
  return { useServicesStore };
});
vi.mock('../stores/settings', () => ({
  useSettingsStore: { getState: () => ({ loadFromBridge: vi.fn() }) },
}));
vi.mock('../i18n', () => ({
  useI18n: () => ({
    lang: 'en',
    t: (key: string, params?: Record<string, string | number>) => Object.entries(params || {})
      .reduce((text, [name, value]) => `${text} ${name}=${value}`, key),
  }),
}));

import { DataSection } from '../components/settings/DataSection';


describe('DataSection maintenance boundary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.chatState.runningSessions = new Set();
    mocks.chatState.status = 'idle';
    mocks.chatState.sessions = [];
    mocks.servicesState.services = [];
    mocks.supportsDataBackupApi.mockResolvedValue(true);
    mocks.fetchServices.mockResolvedValue(undefined);
  });

  it('disables import and export when a session or managed service is running', async () => {
    mocks.chatState.sessions = [
      { id: 'sess-running', title: 'Running', untitled: false, status: 'running' },
    ];
    mocks.servicesState.services = [
      { id: 'reflect/scheduler.py', managed: true, running: true },
      { id: '__bridge__', managed: false, running: true },
    ];

    render(<DataSection />);

    expect((await screen.findByRole('button', { name: 'data.importDataBtn' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: 'data.exportDataBtn' }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole('status').textContent).toContain('sessions=1');
    // The gate names what is running (localized component name), not a bare count.
    expect(screen.getByRole('status').textContent).toContain('services=proc.scheduler');
    expect(mocks.fetchServices).toHaveBeenCalledOnce();
  });

  it('offers a one-click stop for the managed services that hold the gate, and brings them back after the run', async () => {
    mocks.servicesState.services = [{ id: 'frontends/conductor.py', managed: true, running: true }];
    // The real store re-reads the panel before resolving, so the gate is already open when the button settles.
    mocks.stopAllExtras.mockImplementation(async () => { mocks.servicesState.services = []; return true; });
    mocks.startAllExtras.mockResolvedValue(true);
    mocks.tauriInvoke.mockResolvedValue('/data/out.zip');
    mocks.exportData.mockResolvedValue({ path: '/data/out.zip', exportedAt: '', sourceMode: 'included', content: { memory: 0, responses: 0, sessions: 0 } });

    render(<DataSection />);
    fireEvent.click(await screen.findByTestId('data-stop-extras'));
    await waitFor(() => expect(mocks.stopAllExtras).toHaveBeenCalledOnce());
    expect(mocks.toastError).not.toHaveBeenCalled();

    await waitFor(() => expect((screen.getByRole('button', { name: 'data.exportDataBtn' }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'data.exportDataBtn' }));
    await waitFor(() => expect(mocks.modalConfirm).toHaveBeenCalledOnce());
    await act(async () => { await mocks.modalConfirm.mock.calls[0][0].onOk(); });

    expect(mocks.exportData).toHaveBeenCalledOnce();
    expect(mocks.startAllExtras).toHaveBeenCalledOnce();
  });

  it('never restarts extras it did not stop itself', async () => {
    mocks.tauriInvoke.mockResolvedValue('/data/out.zip');
    mocks.exportData.mockResolvedValue({ path: '/data/out.zip', exportedAt: '', sourceMode: 'included', content: { memory: 0, responses: 0, sessions: 0 } });

    render(<DataSection />);
    fireEvent.click(await screen.findByRole('button', { name: 'data.exportDataBtn' }));
    await waitFor(() => expect(mocks.modalConfirm).toHaveBeenCalledOnce());
    await act(async () => { await mocks.modalConfirm.mock.calls[0][0].onOk(); });

    expect(mocks.startAllExtras).not.toHaveBeenCalled();
  });

  it('does not offer the stop when only a chat is running (the user must finish or stop it)', async () => {
    mocks.chatState.status = 'running';

    render(<DataSection />);
    await screen.findByRole('status');
    expect(screen.queryByTestId('data-stop-extras')).toBeNull();
  });

  it('shows backup entrypoints when the bridge reports support', async () => {
    render(<DataSection />);
    expect(screen.getByTestId('data-import-row')).toBeTruthy();
    expect(screen.getByTestId('data-export-row')).toBeTruthy();
  });

  it('keeps backup entrypoints visible while capability support is unknown', async () => {
    mocks.supportsDataBackupApi.mockResolvedValue(null);
    render(<DataSection />);
    expect(screen.getByTestId('data-import-row')).toBeTruthy();
    expect(screen.getByTestId('data-export-row')).toBeTruthy();
    await waitFor(() => expect(mocks.supportsDataBackupApi).toHaveBeenCalledOnce());
    expect(screen.getByTestId('data-import-row')).toBeTruthy();
  });

  it('hides backup entrypoints only when the bridge explicitly reports unsupported', async () => {
    mocks.supportsDataBackupApi.mockResolvedValue(false);
    render(<DataSection />);
    await waitFor(() => expect(screen.queryByTestId('data-import-row')).toBeNull());
    expect(screen.queryByTestId('data-export-row')).toBeNull();
  });

  it('shows exact result statistics, backup path, and recovery guidance', async () => {
    const result = {
      memoryCopied: 3,
      memorySkipped: 0,
      responsesCopied: 4,
      responsesSkipped: 2,
      sessionsAdded: 5,
      sessionsSkipped: 1,
      sessionsFileFound: true,
      backupDir: '/data/temp/memory_import_backup_20260823_120000',
    };
    mocks.tauriInvoke.mockResolvedValue('/data/source.zip');
    mocks.inspectDataImport.mockResolvedValue({
      sourceType: 'backupZip',
      formatVersion: 1,
      exportedAt: '2026-08-23T00:00:00Z',
      sourceMode: 'included',
      content: { memory: 3, responses: 6, sessions: 6 },
    });
    mocks.importData.mockResolvedValue(result);

    render(<DataSection />);
    fireEvent.click(await screen.findByRole('button', { name: 'data.importDataBtn' }));
    fireEvent.click(screen.getByRole('button', { name: 'data.importBackupBtn' }));
    await waitFor(() => expect(mocks.modalConfirm).toHaveBeenCalledOnce());

    await act(async () => {
      await mocks.modalConfirm.mock.calls[0][0].onOk();
    });

    expect(await screen.findByText(result.backupDir)).toBeTruthy();
    expect(screen.getByText(/data.importResultMemoryValue count=3/)).toBeTruthy();
    expect(screen.getByText(/data.importResultAddSkipValue added=4 skipped=2/)).toBeTruthy();
    expect(screen.getByText(/data.importResultAddSkipValue added=5 skipped=1/)).toBeTruthy();
    expect(screen.getByText('data.importRestoreHint')).toBeTruthy();
    expect(mocks.loadSessions).toHaveBeenCalledOnce();
  });
});
