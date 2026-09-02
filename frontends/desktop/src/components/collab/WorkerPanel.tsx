import { useState, useCallback } from 'react';
import { useConductorStore, type Worker } from '../../stores/conductor';
import { useI18n } from '../../i18n';
import { Codicon } from '../../lib/icons';
import { WorkerCard } from './WorkerCard';
import { WorkerDrawer } from './WorkerDrawer';

interface Props {
  collapsed: boolean;
  onToggle: () => void;
}

export function WorkerPanel({ collapsed, onToggle }: Props) {
  const { t } = useI18n();
  const workers = useConductorStore((s) => s.workers);
  const [selectedWorker, setSelectedWorker] = useState<Worker | null>(null);

  const handleCardClick = useCallback((w: Worker) => {
    setSelectedWorker(w);
  }, []);

  const handleCloseDrawer = useCallback(() => {
    setSelectedWorker(null);
  }, []);

  return (
    <>
      <div
        className={`collab-worker-panel ${collapsed ? 'collab-worker-panel--collapsed' : ''}`}
        data-slot="collab-worker-panel"
      >
        <div className="collab-worker-panel-head">
          {!collapsed && (
            <>
              <span className="collab-worker-panel-title">{t('collab.agents')}</span>
              <span className="collab-worker-panel-count">{workers.length}</span>
            </>
          )}
          <button
            className="collab-panel-collapse-btn"
            onClick={onToggle}
            aria-label={t(collapsed ? 'collab.showAgents' : 'collab.hideAgents')}
          >
            <Codicon name={collapsed ? 'layout-sidebar-right-off' : 'layout-sidebar-right'} size="1rem" />
          </button>
        </div>
        {!collapsed && (
          workers.length > 0 ? (
            <div className="collab-worker-list">
              {workers.map((w) => (
                <WorkerCard key={w.id} worker={w} onClick={handleCardClick} />
              ))}
            </div>
          ) : (
            <div className="collab-worker-empty">{t('collab.agentsEmpty')}</div>
          )
        )}
      </div>
      {selectedWorker && (
        <WorkerDrawer worker={selectedWorker} onClose={handleCloseDrawer} />
      )}
    </>
  );
}
