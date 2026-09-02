import { useCallback, useState } from 'react';
import { useI18n } from '../../../i18n';
import { tauriInvoke } from '../../../services/bridge';
import { useChatStore } from '../../../stores/chat';
import { LiveDuration } from '../../layout/LiveDuration';
import { SessionRenameInput } from '../../layout/SessionRenameInput';
import { displayTitle, isPlaceholderTitle } from '../../layout/sessionList';
import { folderName } from '../Composer/workdir';
import './threadHeader.css';

/**
 * Where am I: the active session's title (click to rename), the folder it is bound to (#780, click reveals it)
 * and, while the agent works, how long the current turn has been running (#349).
 */
export function ThreadHeader() {
  const { t } = useI18n();
  const activeSessionId = useChatStore((s) => s.activeSessionId);
  const session = useChatStore((s) => s.sessions.find((item) => item.id === s.activeSessionId));
  const status = useChatStore((s) => s.status);
  const turnStartedAt = useChatStore((s) => s.turnStartedAt);
  const renameSession = useChatStore((s) => s.renameSession);
  const [renaming, setRenaming] = useState(false);

  const workDir = session?.workDir ?? null;
  const reveal = useCallback(() => {
    if (workDir) void tauriInvoke('reveal_in_file_manager', { path: workDir }).catch(() => {});
  }, [workDir]);

  const confirmRename = useCallback((title: string) => {
    if (activeSessionId && title && title !== session?.title) renameSession(activeSessionId, title);
    setRenaming(false);
  }, [activeSessionId, renameSession, session?.title]);

  if (!activeSessionId) return null;

  // The list may lag a freshly created session by one fetch; show the placeholder until it lands.
  const title = session ? displayTitle(session, t) : t('conv.defaultTitle');
  const running = status === 'running' && turnStartedAt != null;

  return (
    <header data-slot="thread-header">
      <div data-slot="thread-header-inner">
        <div data-slot="thread-header-main">
          {renaming ? (
            <SessionRenameInput
              className="thread-header-rename-input"
              initial={session && !isPlaceholderTitle(session.title) ? session.title.trim() : ''}
              onConfirm={confirmRename}
              onCancel={() => setRenaming(false)}
            />
          ) : (
            <button
              type="button"
              data-slot="thread-header-title"
              onClick={() => setRenaming(true)}
              title={t('session.rename')}
            >
              {title}
            </button>
          )}
          {workDir && (
            <button
              type="button"
              data-slot="thread-header-workdir"
              onClick={reveal}
              title={`${t('conv.cwd')}: ${workDir}`}
              aria-label={`${t('conv.cwd')}: ${workDir}`}
            >
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.5 1.5h4.5A1.5 1.5 0 0 1 14 6v6.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 12.5v-8z" stroke="currentColor" strokeWidth="1.2" />
              </svg>
              <span>{folderName(workDir)}</span>
            </button>
          )}
        </div>
        {running && (
          <div data-slot="thread-header-live" role="status" aria-live="off" title={t('status.running')}>
            <span data-slot="thread-header-live-dot" aria-hidden="true" />
            <LiveDuration since={turnStartedAt} />
          </div>
        )}
      </div>
    </header>
  );
}
