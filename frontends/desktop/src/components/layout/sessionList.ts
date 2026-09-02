import type { SessionInfo } from '../../services/chat';

type TFn = (key: string, params?: Record<string, string | number>) => string;

/** Pinned first, then most recently updated. */
export function sortSessions(sessions: readonly SessionInfo[]): SessionInfo[] {
  const stamp = (s: SessionInfo) => Number(s.updatedAt || s.createdAt || 0);
  return [...sessions].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned) || stamp(b) - stamp(a));
}

export function filterSessions(sessions: readonly SessionInfo[], query: string): SessionInfo[] {
  const needle = query.trim().toLowerCase();
  return needle ? sessions.filter((s) => s.title.toLowerCase().includes(needle)) : [...sessions];
}

/** Relative age of a bridge timestamp (Unix seconds) or ISO string, localized. */
export function formatAge(dateVal: number | string | undefined, t: TFn, now = Date.now()): string {
  if (!dateVal) return '';
  const ts = typeof dateVal === 'number' ? dateVal * 1000 : new Date(dateVal).getTime();
  const minutes = Math.floor((now - ts) / 60000);
  if (minutes < 1) return t('conv.age.now');
  if (minutes < 60) return t('conv.age.min', { n: minutes });
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return t('conv.age.hour', { n: hours });
  return t('conv.age.day', { n: Math.floor(hours / 24) });
}

/** The bridge's literal placeholder before the first prompt titles a session. */
const BRIDGE_PLACEHOLDER_TITLE = 'New chat';

export function isPlaceholderTitle(title: string | undefined): boolean {
  const trimmed = title?.trim();
  return !trimmed || trimmed === BRIDGE_PLACEHOLDER_TITLE;
}

/**
 * Title shown for a session row. `untitled` only records that the user never renamed it; the bridge
 * still auto-titles from the first prompt, so only an empty/placeholder title falls back to the localized default.
 */
export function displayTitle(session: Pick<SessionInfo, 'title'>, t: TFn): string {
  return isPlaceholderTitle(session.title) ? t('conv.defaultTitle') : session.title.trim();
}

/** Apply a bridge-announced title to the listed session (identity when nothing changed). */
export function applySessionTitle(
  sessions: SessionInfo[],
  sessionId: string,
  title: string,
  untitled: boolean,
): SessionInfo[] {
  return sessions.map((session) => (
    session.id !== sessionId || (session.title === title && session.untitled === untitled)
      ? session
      : { ...session, title, untitled }
  ));
}
