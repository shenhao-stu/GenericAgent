// @vitest-environment node
import { describe, expect, it } from 'vitest';
import {
  applySessionTitle, displayTitle, filterSessions, formatAge, groupSessions, sortSessions,
} from '../components/layout/sessionList';
import { t } from '../i18n/t';
import type { SessionInfo } from '../services/chat';

const session = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo => ({
  id, title: id, untitled: false, ...extra,
});

describe('sidebar session list', () => {
  it('sorts pinned sessions first, then by most recent activity', () => {
    const sorted = sortSessions([
      session('old', { updatedAt: 100 }),
      session('pinned-old', { pinned: true, updatedAt: 50 }),
      session('new', { updatedAt: 300 }),
      session('created-only', { createdAt: 200 }),
    ]);
    expect(sorted.map((s) => s.id)).toEqual(['pinned-old', 'new', 'created-only', 'old']);
  });

  it('filters by title case-insensitively and ignores surrounding whitespace', () => {
    const sessions = [session('a', { title: 'Weekly Report' }), session('b', { title: 'bug triage' })];
    expect(filterSessions(sessions, '  REPORT ').map((s) => s.id)).toEqual(['a']);
    expect(filterSessions(sessions, '')).toHaveLength(2);
  });

  it('applies a bridge-announced title and clears the placeholder flag', () => {
    const sessions = [session('s1', { title: 'New chat', untitled: true }), session('s2')];
    const next = applySessionTitle(sessions, 's1', '写周报', false);
    expect(next[0]).toEqual({ id: 's1', title: '写周报', untitled: false });
    expect(next[1]).toBe(sessions[1]);
    expect(applySessionTitle(next, 's1', '写周报', false)[0]).toBe(next[0]);
  });

  it('shows a localized placeholder only for empty or bridge-literal titles', () => {
    const zh = (key: string) => t('zh', key);
    expect(displayTitle({ title: 'New chat' }, zh)).toBe('新会话');
    expect(displayTitle({ title: '  ' }, zh)).toBe('新会话');
    expect(displayTitle({ title: '写周报' }, zh)).toBe('写周报');
  });

  it('keeps the auto title from the first prompt even while the user never renamed the session', () => {
    const zh = (key: string) => t('zh', key);
    // The bridge titles from the first prompt but leaves `untitled: true` (= not user-named).
    expect(displayTitle({ title: '请只回复两个字：收到', untitled: true } as SessionInfo, zh)).toBe('请只回复两个字：收到');
  });
});

describe('groupSessions', () => {
  // 2026-09-02 15:30 local time; buckets follow the local calendar day, not 24 h windows.
  const now = new Date(2026, 8, 2, 15, 30).getTime();
  const at = (daysAgo: number, hour: number) => new Date(2026, 8, 2 - daysAgo, hour).getTime() / 1000;

  it('buckets pinned first, then today / yesterday / last 7 days / older, keeping input order inside a bucket', () => {
    const groups = groupSessions(sortSessions([
      session('older', { updatedAt: at(7, 12) }),
      session('week-edge', { updatedAt: at(6, 1) }),
      session('yesterday-late', { updatedAt: at(1, 23) }),
      session('today-early', { updatedAt: at(0, 0) }),
      session('today-now', { updatedAt: at(0, 15) }),
      session('pinned', { pinned: true, updatedAt: at(30, 9) }),
      session('no-stamp'),
    ]), now);
    expect(groups.map((g) => [g.key, g.sessions.map((s) => s.id)])).toEqual([
      ['pinned', ['pinned']],
      ['today', ['today-now', 'today-early']],
      ['yesterday', ['yesterday-late']],
      ['week', ['week-edge']],
      ['older', ['older', 'no-stamp']],
    ]);
  });

  it('omits empty buckets so a single-bucket list needs no labels', () => {
    const groups = groupSessions([session('a', { updatedAt: at(0, 9) }), session('b', { updatedAt: at(0, 8) })], now);
    expect(groups).toHaveLength(1);
    expect(groups[0].key).toBe('today');
    expect(groupSessions([], now)).toEqual([]);
  });

  it('has a dictionary label for every bucket in both languages', () => {
    for (const key of ['pinned', 'today', 'yesterday', 'week', 'older']) {
      expect(t('zh', `conv.group.${key}`)).not.toBe(`conv.group.${key}`);
      expect(t('en', `conv.group.${key}`)).not.toBe(`conv.group.${key}`);
    }
  });
});

describe('formatAge', () => {
  const now = Date.UTC(2026, 8, 2, 12, 0, 0);
  const zh = (key: string, params?: Record<string, string | number>) => t('zh', key, params);
  const en = (key: string, params?: Record<string, string | number>) => t('en', key, params);

  it('localizes relative ages from bridge Unix-second timestamps', () => {
    expect(formatAge(now / 1000 - 30, zh, now)).toBe('刚刚');
    expect(formatAge(now / 1000 - 5 * 60, zh, now)).toBe('5 分钟前');
    expect(formatAge(now / 1000 - 3 * 3600, en, now)).toBe('3h');
    expect(formatAge(now / 1000 - 2 * 86400, en, now)).toBe('2d');
  });

  it('returns an empty string without a timestamp', () => {
    expect(formatAge(undefined, en, now)).toBe('');
  });
});
