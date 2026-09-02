/**
 * Whether the user can see what just changed: the document is visible and the window has focus.
 * Hidden-to-tray (Windows close), minimized and background windows all count as unattended.
 */
export function windowIsAttended(): boolean {
  return document.visibilityState === 'visible' && document.hasFocus();
}

/** Tauri's `UserAttentionType.Informational` (taskbar flash / dock bounce once); no-op outside the shell. */
export function requestUserAttention(): void {
  const tauri = (window as any).__TAURI__?.window;
  const current = tauri?.getCurrentWindow?.();
  if (!current?.requestUserAttention) return;
  const informational = tauri?.UserAttentionType?.Informational ?? 2;
  void Promise.resolve(current.requestUserAttention(informational)).catch(() => {});
}

/** Fires when the window (re)gains the user's attention. Returns the unsubscribe. */
export function onWindowAttended(handler: () => void): () => void {
  const onVisibility = () => { if (windowIsAttended()) handler(); };
  window.addEventListener('focus', handler);
  document.addEventListener('visibilitychange', onVisibility);
  return () => {
    window.removeEventListener('focus', handler);
    document.removeEventListener('visibilitychange', onVisibility);
  };
}
