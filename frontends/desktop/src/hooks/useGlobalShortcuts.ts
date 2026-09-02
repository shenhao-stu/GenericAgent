import { useEffect } from 'react';
import { isMacOS } from '../platform';
import { useAppStore } from '../stores/app';
import { useChatStore } from '../stores/chat';
import { useSettingsStore } from '../stores/settings';

/** App-wide shortcuts: one table, one listener. Modifier = Cmd on macOS, Ctrl elsewhere. */
export const GLOBAL_SHORTCUTS = [
  { key: 'b', run: () => useAppStore.getState().toggleSidebar() },
  { key: 'n', run: () => { useChatStore.getState().newSession(); useAppStore.getState().setPage('chat'); } },
  { key: ',', run: () => useSettingsStore.getState().open() },
] as const;

export const MODIFIER_LABEL = isMacOS ? '⌘' : 'Ctrl+';

export function shortcutFor(key: string): string {
  return `${MODIFIER_LABEL}${key.toUpperCase()}`;
}

export function matchShortcut(event: KeyboardEvent) {
  const modifier = isMacOS ? event.metaKey : event.ctrlKey;
  if (!modifier || event.altKey || event.shiftKey) return undefined;
  return GLOBAL_SHORTCUTS.find((shortcut) => shortcut.key === event.key.toLowerCase());
}

export function useGlobalShortcuts() {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const shortcut = matchShortcut(event);
      if (!shortcut) return;
      event.preventDefault();
      shortcut.run();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}
