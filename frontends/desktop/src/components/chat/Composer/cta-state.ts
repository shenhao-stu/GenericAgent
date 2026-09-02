export type CTAState = 'send' | 'stop' | 'queue' | 'disabled';

export function computeCTAState(
  isGenerating: boolean,
  hasContent: boolean,
  hasPendingUploads: boolean = false,
): CTAState {
  if (hasPendingUploads) return 'disabled';
  if (isGenerating && hasContent) return 'queue';
  if (isGenerating) return 'stop';
  if (hasContent) return 'send';
  return 'disabled';
}
