import type { Worker } from '../../stores/conductor';
import { useI18n } from '../../i18n';
import { MarkdownPart } from '../chat/Thread/parts/MarkdownPart';

interface Props {
  worker: Worker;
  onClose: () => void;
}

export function WorkerDrawer({ worker, onClose }: Props) {
  const { t } = useI18n();
  return (
    <div className="collab-drawer-wrap" data-slot="collab-drawer-wrap">
      <div className="collab-drawer-backdrop" onClick={onClose} />
      <aside className="collab-drawer" data-slot="collab-drawer">
        <div className="collab-drawer-head">
          <span className="collab-drawer-title">{worker.title}</span>
          <button className="collab-drawer-close" onClick={onClose} aria-label={t('common.close')}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
              <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <div className="collab-drawer-body">
          {worker.fullReply ? (
            <MarkdownPart content={worker.fullReply} />
          ) : (
            <p className="collab-drawer-empty">{t('collab.summaryWait')}</p>
          )}
        </div>
      </aside>
    </div>
  );
}
