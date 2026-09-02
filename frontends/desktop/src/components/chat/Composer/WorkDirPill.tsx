import { useCallback } from 'react';
import { useI18n } from '../../../i18n';
import { tauriInvoke } from '../../../services/bridge';
import { useChatStore } from '../../../stores/chat';
import { folderName } from './workdir';

/**
 * Picks the folder the next new session will be bound to (#780). Once a session exists its folder is
 * shown by the thread header, so the pill only appears before the first message.
 */
export function WorkDirPill() {
  const { t } = useI18n();
  const activeSessionId = useChatStore((s) => s.activeSessionId);
  const pendingWorkDir = useChatStore((s) => s.pendingWorkDir);
  const setPendingWorkDir = useChatStore((s) => s.setPendingWorkDir);
  const inTauri = typeof (window as any).__TAURI__?.core?.invoke === 'function';

  const pick = useCallback(async () => {
    const picked = await tauriInvoke('pick_directory', { title: t('conv.cwdPickerTitle') }) as string | null;
    if (picked) setPendingWorkDir(picked);
  }, [setPendingWorkDir, t]);

  if (!inTauri || activeSessionId) return null;

  const label = pendingWorkDir ? folderName(pendingWorkDir) : t('conv.cwdPick');
  const tip = pendingWorkDir ? `${t('conv.cwd')}: ${pendingWorkDir}` : t('conv.cwdHint');

  return (
    <div data-slot="workdir-pill" data-set={pendingWorkDir ? '' : undefined}>
      <button
        type="button"
        data-slot="workdir-pill-btn"
        onClick={pick}
        aria-label={tip}
        title={tip}
      >
        <FolderIcon />
        <span data-slot="workdir-pill-label">{label}</span>
      </button>
      {pendingWorkDir && (
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
