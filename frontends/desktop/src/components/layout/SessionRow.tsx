import { useState, useCallback } from 'react';
import { Dropdown, Modal } from '@douyinfe/semi-ui';
import type { SessionInfo } from '../../services/chat';
import { useChatStore } from '../../stores/chat';
import { useI18n } from '../../i18n';
import { Codicon } from '../../lib/icons';
import { LiveDuration } from './LiveDuration';
import { SessionRenameInput } from './SessionRenameInput';
import { displayTitle, formatAge, isPlaceholderTitle } from './sessionList';

export function SessionRow({
  session,
  isActive,
  isWorking,
  isUnread,
  onClick,
}: {
  session: SessionInfo;
  isActive: boolean;
  isWorking?: boolean;
  /** A turn finished here while the user was looking elsewhere; cleared when the session is opened. */
  isUnread?: boolean;
  onClick: () => void;
}) {
  const { t } = useI18n();
  const renameSession = useChatStore((s) => s.renameSession);
  const deleteSession = useChatStore((s) => s.deleteSession);
  const pinSession = useChatStore((s) => s.pinSession);
  const turnStartedAt = useChatStore((s) => s.sessionsById[session.id]?.turnStartedAt ?? null);

  const [menuOpen, setMenuOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);

  const handleRenameStart = useCallback(() => {
    setRenaming(true);
    setMenuOpen(false);
  }, []);

  const handleRenameConfirm = useCallback((title: string) => {
    if (title && title !== session.title) {
      renameSession(session.id, title);
    }
    setRenaming(false);
  }, [session.id, session.title, renameSession]);

  const handleRenameCancel = useCallback(() => {
    setRenaming(false);
  }, []);

  const handlePin = useCallback(() => {
    setMenuOpen(false);
    setTimeout(() => pinSession(session.id, !session.pinned), 0);
  }, [session.id, session.pinned, pinSession]);

  const handleDelete = useCallback(() => {
    setMenuOpen(false);
    setTimeout(() => {
      Modal.confirm({
        title: t('session.delete'),
        content: t('session.deleteConfirm'),
        okType: 'danger',
        onOk: () => deleteSession(session.id),
      });
    }, 0);
  }, [session.id, deleteSession, t]);

  const menu = (
    <Dropdown.Menu className="ga-session-menu">
      <Dropdown.Item onClick={handleRenameStart}>
        <Codicon name="edit" size="0.875rem" />
        <span>{t('session.rename')}</span>
      </Dropdown.Item>
      <Dropdown.Item onClick={handlePin}>
        <Codicon name="pin" size="0.875rem" />
        <span>{session.pinned ? t('session.unpin') : t('session.pin')}</span>
      </Dropdown.Item>
      <Dropdown.Item type="danger" onClick={handleDelete}>
        <Codicon name="trash" size="0.875rem" />
        <span>{t('session.delete')}</span>
      </Dropdown.Item>
    </Dropdown.Menu>
  );

  return (
    <div
      className={`ga-session-item${isActive ? ' active' : ''}${isUnread ? ' unread' : ''}`}
      data-session-id={session.id}
      data-unread={isUnread ? '' : undefined}
      onClick={renaming || menuOpen ? undefined : onClick}
      onContextMenu={(e) => { e.preventDefault(); if (!renaming) setMenuOpen(true); }}
    >
      <span className="ga-session-content">
        <span
          className={`ga-status-dot${isWorking ? ' working' : isUnread ? ' unread' : ''}`}
          title={isUnread && !isWorking ? t('conv.unread') : undefined}
        />
        {renaming ? (
          <SessionRenameInput
            className="ga-session-rename-input"
            initial={isPlaceholderTitle(session.title) ? '' : session.title.trim()}
            onConfirm={handleRenameConfirm}
            onCancel={handleRenameCancel}
          />
        ) : (
          <span className="ga-session-title" title={session.cwd || undefined}>
            {displayTitle(session, t)}
          </span>
        )}
      </span>

      {!renaming && (
        <>
          {session.pinned && (
            <span className="ga-session-pin-icon">
              <Codicon name="pinned" size="0.875rem" />
            </span>
          )}
          {isWorking && turnStartedAt ? (
            <span className="ga-session-age ga-session-duration">
              <LiveDuration since={turnStartedAt} />
            </span>
          ) : (
            <span className="ga-session-age">{formatAge(session.updatedAt, t)}</span>
          )}
          <span
            className={`ga-session-actions${menuOpen ? ' menu-open' : ''}`}
            onClick={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <Dropdown
              trigger="click"
              position="bottomRight"
              visible={menuOpen}
              onVisibleChange={setMenuOpen}
              render={menu}
            >
              <button
                type="button"
                className="ga-session-actions-btn"
                onClick={(e) => e.stopPropagation()}
                aria-label={t('conv.actions')}
              >
                <Codicon name="kebab-vertical" size="0.875rem" />
              </button>
            </Dropdown>
          </span>
        </>
      )}
    </div>
  );
}
