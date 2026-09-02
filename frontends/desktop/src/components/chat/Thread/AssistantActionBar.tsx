import { memo, useCallback, useRef, useState } from 'react';
import { useI18n } from '../../../i18n';
import { formatClock, formatDuration } from '../../../utils/format';
import './AssistantActionBar.css';

interface Props {
  getMessageText: () => string;
  executionMs?: number;
  /** Completion time in ms since epoch (bridge `ts` of the final message). */
  finishedAt?: number;
}

export const AssistantActionBar = memo(function AssistantActionBar({ getMessageText, executionMs, finishedAt }: Props) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const handleCopy = useCallback(() => {
    const text = getMessageText();
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => setCopied(false), 2000);
    });
  }, [getMessageText]);

  if (typeof navigator === 'undefined' || !navigator.clipboard) return null;

  return (
    <div data-slot="assistant-action-bar">
      <button
        data-slot="action-bar-btn"
        onClick={handleCopy}
        title={t(copied ? 'act.copied' : 'act.copy')}
        aria-label={t(copied ? 'act.copied' : 'act.copy')}
      >
        {copied ? (
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M3 8.5l3 3 7-7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <rect x="5.5" y="5.5" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.2" />
            <path d="M3.5 10.5v-7a1 1 0 011-1h7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
        )}
      </button>
      {typeof executionMs === 'number' && executionMs > 0 && (
        <span data-slot="action-bar-duration" title={t('msg.duration')}>
          {formatDuration(executionMs)}
        </span>
      )}
      {typeof finishedAt === 'number' && finishedAt > 0 && (
        <time
          data-slot="action-bar-time"
          dateTime={new Date(finishedAt).toISOString()}
          title={t('msg.finishedAt', { time: new Date(finishedAt).toLocaleString() })}
        >
          {formatClock(finishedAt)}
        </time>
      )}
    </div>
  );
});
