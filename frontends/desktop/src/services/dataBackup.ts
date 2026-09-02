import { BRIDGE_BASE } from './constants';
import { BridgeError, postJson } from './http';

export type BackupSourceMode = 'included' | 'localRepository';

export interface BackupContentCounts {
  memory: number;
  responses: number;
  sessions: number;
}

export interface BackupInspection {
  sourceType: 'backupZip' | 'legacyFolder';
  formatVersion: number | null;
  exportedAt: string | null;
  sourceMode: BackupSourceMode | null;
  content: BackupContentCounts;
}

export interface DataImportResult {
  memoryCopied: number;
  memorySkipped: number;
  responsesCopied: number;
  responsesSkipped: number;
  sessionsAdded: number;
  sessionsSkipped: number;
  sessionsFileFound: boolean;
  backupDir: string;
}

export interface DataExportResult {
  path: string;
  exportedAt: string;
  sourceMode: BackupSourceMode;
  content: BackupContentCounts;
}

export type DataBackupAvailability = true | false | null;

const stringList = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];

/** A `maintenance_conflict` (409) names what is still running so the UI can say exactly what to stop. */
export class DataBackupError extends BridgeError {
  readonly runningSessions = stringList(this.payload.runningSessions);
  readonly runningExtras = stringList(this.payload.runningExtras);
}

const asBackupError = (error: unknown): never => {
  throw error instanceof BridgeError ? new DataBackupError(error.message, error.status, error.payload) : error;
};

const post = <T>(path: string, body: Record<string, unknown>): Promise<T> => postJson<T>(path, body).catch(asBackupError);

export async function supportsDataBackupApi(): Promise<DataBackupAvailability> {
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort(), 3_000);
  try {
    const response = await fetch(`${BRIDGE_BASE}/services/capabilities`, {
      method: 'GET',
      signal: controller.signal,
    });
    if (response.status === 404) return false;
    if (!response.ok) return null;
    const payload: unknown = await response.json();
    if (
      typeof payload !== 'object'
      || payload === null
      || typeof (payload as { dataBackup?: unknown }).dataBackup !== 'boolean'
    ) {
      return null;
    }
    return (payload as { dataBackup: boolean }).dataBackup;
  } catch {
    return null;
  } finally {
    globalThis.clearTimeout(timeout);
  }
}

export function backupFilename(lang: string, date = new Date()): string {
  const parts = {
    year: date.getFullYear(),
    month: String(date.getMonth() + 1).padStart(2, '0'),
    day: String(date.getDate()).padStart(2, '0'),
    hour: String(date.getHours()).padStart(2, '0'),
    minute: String(date.getMinutes()).padStart(2, '0'),
    second: String(date.getSeconds()).padStart(2, '0'),
  };
  const stamp = `${parts.year}-${parts.month}-${parts.day}-${parts.hour}${parts.minute}${parts.second}`;
  const label = lang === 'zh' ? '数据备份' : 'data-backup';
  return `GenericAgent-${label}-${stamp}.zip`;
}

export const inspectDataImport = (sourcePath: string) => post<BackupInspection>('/memory/import/inspect', { sourcePath });
export const importData = (sourcePath: string) => post<DataImportResult>('/memory/import', { sourcePath });
export const exportData = (destinationPath: string, sourceMode: BackupSourceMode) =>
  post<DataExportResult>('/memory/export', { destinationPath, sourceMode });
