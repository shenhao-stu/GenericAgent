import { useCallback } from 'react';
import { useI18n } from '../../../i18n';
import { tauriInvoke } from '../../../services/bridge';
import { useChatStore } from '../../../stores/chat';
import { folderName } from './workdir';

/**
 * Session working directory (#780). Before the first message it picks the folder the new session
 * will be bound to; once the session exists it shows the bound folder and reveals it on click.
 */
export function WorkDirPill() {
  const { t } = useI18n();
  const activeSessionId = useChatStore((s) => s.activeSessionId);
  const pendingWorkDir = useChatStore((s) => s.pendingWorkDir);
  const setPendingWorkDir = useChatStore((s) => s.setPendingWorkDir);
  const boundWorkDir = useChatStore((s) => s.sessions.find((session) => session.id === activeSessionId)?.workDir ?? null);
  const inTauri = typeof (window as any).__TAURI__?.core?.invoke === 'function';

  const pick = useCallback(async () => {
    const picked = await tauriInvoke('pick_directory', { title: t('conv.cwdPickerTitle') }) as string | null;
    if (picked) setPendingWorkDir(picked);
  }, [setPendingWorkDir, t]);

  const reveal = useCallback(() => {
    if (boundWorkDir) void tauriInvoke('reveal_in_file_manager', { path: boundWorkDir }).catch(() => {});
  }, [boundWorkDir]);

  if (!inTauri) return null;
  if (activeSessionId && !boundWorkDir) return null;

  const dir = activeSessionId ? boundWorkDir! : pendingWorkDir;
  const label = dir ? folderName(dir) : t('conv.cwdPick');
  const tip = dir ? `${t('conv.cwd')}: ${dir}` : t('conv.cwdHint');

  return (
    <div data-slot="workdir-pill" data-bound={activeSessionId ? '' : undefined} data-set={dir ? '' : undefined}>
      <button
        type="button"
        data-slot="workdir-pill-btn"
        onClick={activeSessionId ? reveal : pick}
        aria-label={tip}
        title={tip}
      >
        <FolderIcon />
        <span data-slot="workdir-pill-label">{label}</span>
      </button>
      {!activeSessionId && pendingWorkDir && (
        <button
          type="button"
          data-slot="workdir-pill-clear"
          onClick={() => setPendingWorkDir(null)}
          aria-label={t('conv.cwdDefault')}
          title={t('conv.cwdDefault')}
        >
          ×
        </button>
      )}
    </div>
  );
}

function FolderIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.5 1.5h4.5A1.5 1.5 0 0 1 14 6v6.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 12.5v-8z" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}
