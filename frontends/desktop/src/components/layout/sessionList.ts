import type { SessionInfo } from '../../services/chat';

type TFn = (key: string, params?: Record<string, string | number>) => string;

/** Bridge stamps are Unix seconds (or, defensively, ISO strings); the UI works in ms. */
function stampMs(value: number | string | undefined): number {
  if (!value) return 0;
  return typeof value === 'number' ? value * 1000 : new Date(value).getTime();
}

const activityMs = (s: SessionInfo) => stampMs(s.updatedAt || s.createdAt);

/** Pinned first, then most recently updated. */
export function sortSessions(sessions: readonly SessionInfo[]): SessionInfo[] {
  return [...sessions].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned) || activityMs(b) - activityMs(a));
}

export function filterSessions(sessions: readonly SessionInfo[], query: string): SessionInfo[] {
  const needle = query.trim().toLowerCase();
  return needle ? sessions.filter((s) => s.title.toLowerCase().includes(needle)) : [...sessions];
}

export type SessionGroupKey = 'pinned' | 'today' | 'yesterday' | 'week' | 'older';
export const SESSION_GROUP_ORDER: readonly SessionGroupKey[] = ['pinned', 'today', 'yesterday', 'week', 'older'];

function localDayIndex(ms: number): number {
  const d = new Date(ms);
  return Math.floor((d.getTime() - d.getTimezoneOffset() * 60000) / 86400000);
}

function groupKey(s: SessionInfo, todayIndex: number): SessionGroupKey {
  if (s.pinned) return 'pinned';
  const age = todayIndex - localDayIndex(activityMs(s));
  if (age <= 0) return 'today';
  if (age === 1) return 'yesterday';
  return age < 7 ? 'week' : 'older';
}

/** Sorted sessions bucketed by local calendar age; empty buckets are dropped so the sidebar never shows a bare label. */
export function groupSessions(sessions: readonly SessionInfo[], now = Date.now()): { key: SessionGroupKey; items: SessionInfo[] }[] {
  const todayIndex = localDayIndex(now);
  const buckets = new Map<SessionGroupKey, SessionInfo[]>();
  for (const s of sortSessions(sessions)) {
    const key = groupKey(s, todayIndex);
    buckets.set(key, [...(buckets.get(key) ?? []), s]);
  }
  return SESSION_GROUP_ORDER.flatMap((key) => (buckets.has(key) ? [{ key, items: buckets.get(key)! }] : []));
}

/** Relative age of a bridge timestamp (Unix seconds) or ISO string, localized. */
export function formatAge(dateVal: number | string | undefined, t: TFn, now = Date.now()): string {
  if (!dateVal) return '';
  const ts = stampMs(dateVal);
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
