import { CONDUCTOR_BASE } from './constants';
import { getJson } from './http';

export interface TokenRecord {
  thread: string;
  input: number;
  output: number;
  cacheCreate: number;
  cacheRead: number;
  model: string;
}

export interface HistoryEntry {
  id: string;
  title: string;
  input: number;
  output: number;
  cacheWrite: number;
  cacheRead: number;
  model: string;
  ts: number;
  deleted?: boolean;
}

export interface TokenSnapshot {
  totalInput: number;
  totalOutput: number;
  totalCacheWrite: number;
  totalCacheRead: number;
}

export function emptySnapshot(): TokenSnapshot {
  return { totalInput: 0, totalOutput: 0, totalCacheWrite: 0, totalCacheRead: 0 };
}

export interface TokenHistoryResponse {
  history: HistoryEntry[];
  snap: TokenSnapshot;
  conductorSnapshot?: TokenSnapshot;
}

export async function fetchTokenHistory(): Promise<TokenHistoryResponse> {
  const data = await getJson('/token-history');

  const rawHistory: Array<Record<string, unknown>> = data.history ?? [];
  const history: HistoryEntry[] = rawHistory.map((r) => {
    const rawTs = Number(r.ts ?? 0);
    return {
      id: String(r.sessionId ?? r.id ?? ''),
      title: String(r.title ?? ''),
      input: Number(r.input ?? 0),
      output: Number(r.output ?? 0),
      cacheWrite: Number(r.cacheCreate ?? r.cacheWrite ?? 0),
      cacheRead: Number(r.cacheRead ?? 0),
      model: String(r.model ?? ''),
      ts: rawTs < 1e12 ? rawTs * 1000 : rawTs,
      deleted: r.deleted as boolean | undefined,
    };
  });

  // Backend snap is a per-session dict; aggregate into a single snapshot
  const rawSnap: Record<string, Record<string, number>> = data.snap ?? {};
  const snap: TokenSnapshot = Object.values(rawSnap).reduce(
    (acc, s) => ({
      totalInput: acc.totalInput + (s.input ?? 0),
      totalOutput: acc.totalOutput + (s.output ?? 0),
      totalCacheWrite: acc.totalCacheWrite + (s.cacheCreate ?? s.cacheWrite ?? 0),
      totalCacheRead: acc.totalCacheRead + (s.cacheRead ?? 0),
    }),
    emptySnapshot(),
  );

  // conductorHist and conductorLast are aggregate stats objects, not arrays
  const rawConductorLast = data.conductorLast as Record<string, number> | undefined;
  const conductorSnapshot: TokenSnapshot | undefined = rawConductorLast
    ? {
        totalInput: rawConductorLast.input ?? 0,
        totalOutput: rawConductorLast.output ?? 0,
        totalCacheWrite: rawConductorLast.cacheCreate ?? 0,
        totalCacheRead: rawConductorLast.cacheRead ?? 0,
      }
    : undefined;

  return {
    history,
    snap,
    conductorSnapshot,
  };
}

type Records = { records?: TokenRecord[] };
export const fetchConductorTokenStats = () => getJson<Records>('/token-stats', CONDUCTOR_BASE).then((d) => d.records ?? []);
export const fetchLiveTokenStats = () => getJson<Records>('/token-stats').then((d) => d.records ?? []);
