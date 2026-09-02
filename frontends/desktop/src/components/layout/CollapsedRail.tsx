import { useCallback, useState } from 'react';
import { Tooltip } from '@douyinfe/semi-ui';
import { Codicon } from '../../lib/icons';
import { useAppStore } from '../../stores/app';
import { useChatStore } from '../../stores/chat';
import { useSettingsStore } from '../../stores/settings';
import { useI18n } from '../../i18n';
import { shortcutFor } from '../../hooks/useGlobalShortcuts';
import { NAV_ITEMS } from './navItems';
import { BridgeMenuPanel } from './BridgeMenuPanel';

/**
 * Icon-only navigation shown while the sidebar is collapsed, so hiding the session list never hides the app:
 * new chat (with an unread dot), the pages, settings and runtime management stay one click away.
 */
export function CollapsedRail() {
  const { activePage, setPage } = useAppStore();
  const openSettings = useSettingsStore((s) => s.open);
  const newSession = useChatStore((s) => s.newSession);
  const hasUnread = useChatStore((s) => s.unreadSessions.size > 0);
  const { t } = useI18n();
  const [runtimePanelOpen, setRuntimePanelOpen] = useState(false);
  const newSessionLabel = t('nav.chatShortcut', { shortcut: shortcutFor('n') });

  const closeRuntimePanel = useCallback(() => setRuntimePanelOpen(false), []);

  const handleNewSession = useCallback(() => {
    newSession();
    setPage('chat');
  }, [newSession, setPage]);

  const handleOpenSettings = useCallback(() => {
    closeRuntimePanel();
    openSettings();
  }, [closeRuntimePanel, openSettings]);

  return (
    <nav className="ga-collapsed-rail" aria-label={t('nav.main')} data-slot="collapsed-rail">
      <Tooltip content={newSessionLabel} position="right">
        <button
          type="button"
          className={`ga-rail-btn${activePage === 'chat' ? ' active' : ''}`}
          onClick={handleNewSession}
          aria-label={newSessionLabel}
          aria-current={activePage === 'chat' ? 'page' : undefined}
        >
          <Codicon name="comment" size="1rem" />
          {hasUnread && <span className="ga-rail-badge" title={t('conv.unread')} />}
        </button>
      </Tooltip>
      {NAV_ITEMS.map((item) => (
        <Tooltip key={item.key} content={t(item.textKey)} position="right">
          <button
            type="button"
            className={`ga-rail-btn${activePage === item.key ? ' active' : ''}`}
            onClick={() => setPage(item.key)}
            aria-label={t(item.textKey)}
            aria-current={activePage === item.key ? 'page' : undefined}
          >
            <Codicon name={item.icon} size="1rem" />
          </button>
        </Tooltip>
      ))}
      <div className="ga-rail-spacer" />
      <Tooltip content={t('foot.settings')} position="right">
        <button type="button" className="ga-rail-btn" onClick={handleOpenSettings} aria-label={t('foot.settings')}>
          <Codicon name="settings-gear" size="1rem" />
        </button>
      </Tooltip>
      <div className="ga-sidebar-footer-anchor">
        <Tooltip content={t('foot.runtimeManagement')} position="right">
          <button
            type="button"
            className="ga-rail-btn"
            onClick={() => setRuntimePanelOpen((open) => !open)}
            aria-label={t('foot.runtimeManagement')}
            aria-expanded={runtimePanelOpen}
            aria-haspopup="dialog"
            aria-controls="ga-runtime-management-panel"
          >
            <Codicon name="server-process" size="1rem" />
          </button>
        </Tooltip>
        {runtimePanelOpen && <BridgeMenuPanel onClose={closeRuntimePanel} />}
      </div>
    </nav>
  );
}
