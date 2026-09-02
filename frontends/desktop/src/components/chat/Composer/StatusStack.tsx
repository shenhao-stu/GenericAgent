import { useChatStore } from '../../../stores/chat';
import { useI18n } from '../../../i18n';

export function StatusStack() {
  const { t } = useI18n();
  const isGenerating = useChatStore((s) => s.status === 'running');
  const queue = useChatStore((s) => s.pendingQueue);
  const cancelQueued = useChatStore((s) => s.cancelQueued);

  if (!isGenerating && queue.length === 0) return null;

  return (
    <div data-slot="composer-status-stack">
      {isGenerating && (
        <div data-slot="status-running">
          <span data-slot="status-dot" />
          <span data-slot="status-label">{t('composer.thinking')}</span>
        </div>
      )}
      {queue.map((item, i) => (
        <div key={i} data-slot="status-queued">
          <span data-slot="status-queue-num">#{i + 1}</span>
          <span data-slot="status-queue-text">{item.text.slice(0, 40)}{item.text.length > 40 ? '…' : ''}</span>
          <button data-slot="status-queue-cancel" onClick={() => cancelQueued(i)} aria-label={t('composer.cancelQueued')}>
            <svg width="10" height="10" viewBox="0 0 16 16" fill="none">
              <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
}
