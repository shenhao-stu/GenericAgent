import { useConductorStore, type Worker } from '../../stores/conductor';
import { useI18n } from '../../i18n';

const BADGES: { status: Worker['status']; tone: 'running' | 'done' | 'issue'; slot: string; labelKey: string }[] = [
  { status: 'running', tone: 'running', slot: 'collab-rail-run', labelKey: 'collab.railActive' },
  { status: 'reported', tone: 'done', slot: 'collab-rail-done', labelKey: 'collab.railReported' },
  { status: 'failed', tone: 'issue', slot: 'collab-rail-issue', labelKey: 'collab.railFailed' },
];

export function WorkerRail() {
  const { t } = useI18n();
  const workers = useConductorStore((s) => s.workers);

  return (
    <div className="collab-rail" data-slot="collab-rail">
      {BADGES.map(({ status, tone, slot, labelKey }) => {
        const count = workers.filter((w) => w.status === status).length;
        if (count === 0) return null;
        return (
          <span key={status} className={`collab-rail-badge collab-rail-badge--${tone}`} data-slot={slot}>
            <span className={`collab-rail-dot collab-rail-dot--${tone}`} />
            <span className="collab-rail-n">{count}</span>
            <span className="collab-rail-label">{t(labelKey)}</span>
          </span>
        );
      })}
    </div>
  );
}
