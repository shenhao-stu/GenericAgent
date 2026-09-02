export type CTAState = 'send' | 'stop' | 'busy' | 'queue' | 'disabled';

/**
 * The single button at the end of the composer. `busy` replaces `stop` when the backend cannot interrupt a
 * running turn (conductor): a Stop that does nothing is worse than an honest disabled indicator.
 */
export function computeCTAState(
  isGenerating: boolean,
  hasContent: boolean,
  hasPendingUploads: boolean = false,
  canStop: boolean = true,
): CTAState {
  if (hasPendingUploads) return 'disabled';
  if (isGenerating && hasContent) return 'queue';
  if (isGenerating) return canStop ? 'stop' : 'busy';
  if (hasContent) return 'send';
  return 'disabled';
}
