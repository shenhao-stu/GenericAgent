import { memo, useRef } from 'react';
import { useI18n } from '../../../../i18n';
import { useEnterAnimation } from '../../../../hooks/useEnterAnimation';
import { useSegmentDisclosure } from '../../../../stores/thread-view';

interface Props {
  sessionId: string;
  content: string;
  inFlight: boolean;
  segmentKey?: string;
  isStreaming?: boolean;
}

export const ResultPart = memo(function ResultPart({ sessionId, content, inFlight, segmentKey = '', isStreaming = false }: Props) {
  const { t } = useI18n();
  const { expanded, setExpanded } = useSegmentDisclosure(
    sessionId,
    segmentKey,
    inFlight || isStreaming,
  );
  const isLong = content.length > 200;
  const ref = useRef<HTMLDivElement>(null);
  useEnterAnimation(ref, segmentKey, isStreaming);

  return (
    <div ref={ref} data-slot="tool-block" data-kind="result" data-tool-row data-status={inFlight ? 'running' : 'success'}>
      <div data-slot="tool-header" onClick={() => isLong && setExpanded(!expanded)}>
        <span data-slot="tool-title">{t(inFlight ? 'fold.outputRunning' : 'fold.output')}</span>
        {isLong && !expanded && (
          <span data-slot="tool-duration">{t('fold.chars', { n: content.length })}</span>
        )}
      </div>
      {(expanded || !isLong) && (
        <pre data-slot="tool-body">{content}</pre>
      )}
    </div>
  );
});
