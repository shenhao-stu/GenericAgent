import { deleteJson, getJson, patchJson, postJson } from './http';

export type MessageStatus = 'completed' | 'in_progress' | 'failed';

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'error';
  content: string;
  status: MessageStatus;
  createdAt?: number;
  ts?: number;
  turn_segs?: string[];
  images?: { name: string; path: string }[];
  files?: { name: string; path: string; size?: number }[];
  executionMs?: number;
}

export interface SessionInfo {
  id: string;
  title: string;
  untitled: boolean;
  pinned?: boolean;
  status?: 'idle' | 'running' | 'error' | 'cancelled';
  updatedAt?: number | string;
  createdAt?: number | string;
  /** Directory recorded by the bridge (its GA root unless the session was bound to a folder). */
  cwd?: string;
  /** Folder the session was explicitly bound to; null/undefined means the agent's default. */
  workDir?: string | null;
}

export interface LiveModel {
  isMixin: boolean;
  current: string;
  llmNo?: number;
  runningLlmNo?: number | null;
  runningModel?: string | null;
}

export interface PollResult {
  messages: Message[];
  partial?: Message;
  status: 'running' | 'idle';
  plan?: unknown;
  model?: LiveModel;
}

export interface DropStat {
  isDir: boolean;
  size: number;
  name: string;
  preview?: string;
}

type Attachment = { name: string; path: string; size?: number };
type ImageAttachment = { name: string; path: string; base64?: string };

function normalizeMessage(msg: Record<string, unknown>, status: MessageStatus = 'completed'): Message {
  // Bridge timestamps (`ts`, `createdAt`) are Unix seconds; the UI works in ms. Local optimistic
  // messages are already ms and never pass through here. `executionMs` is already ms.
  const rawTs = (msg.createdAt as number) ?? (msg.ts as number);
  return {
    id: String(msg.id),
    role: msg.role as Message['role'],
    content: (msg.content as string) || '',
    status: (msg.status as MessageStatus) ?? status,
    createdAt: typeof rawTs === 'number' ? Math.round(rawTs * 1000) : undefined,
    ...(Array.isArray(msg.turn_segs) && { turn_segs: msg.turn_segs as string[] }),
    ...(Array.isArray(msg.images) && msg.images.length > 0 && { images: msg.images as Message['images'] }),
    ...(Array.isArray(msg.files) && msg.files.length > 0 && { files: msg.files as Message['files'] }),
    ...(typeof msg.executionMs === 'number' && { executionMs: msg.executionMs }),
  };
}

export async function createSession(cwd = ''): Promise<string> {
  const data = await postJson('/session/new', { cwd, mcp_servers: [] });
  if (typeof data.sessionId !== 'string' || !data.sessionId) throw new Error('bridge returned no sessionId');
  return data.sessionId;
}

/** Uploads land under desktop_uploads/<sid>/ so `/upload/raw` can serve them back into the thread. */
async function uploadDataUrl(sid: string, name: string, dataUrl: string): Promise<string> {
  return (await postJson('/upload', { name, dataUrl, sid })).path;
}

export const uploadFile = (name: string, dataUrl: string) => uploadDataUrl('_files', name, dataUrl);

/**
 * Inspect a path dropped onto the window via Tauri's native drag-drop. Native drops carry absolute
 * paths, so the bridge says whether it is a folder or file and — for images with `preview` — returns a
 * base64 thumbnail. Files/folders otherwise reach the agent by path; no bytes cross for them.
 */
export function statDroppedPath(path: string, preview: boolean): Promise<DropStat | null> {
  return postJson('/drop/stat', { path, preview })
    .then((d) => ({ isDir: !!d.is_dir, size: d.size ?? 0, name: d.name ?? path, preview: d.preview }))
    .catch(() => null);
}

export async function sendPrompt(
  sessionId: string,
  prompt: string,
  files?: Attachment[],
  images?: ImageAttachment[],
): Promise<string> {
  const filesMeta = (files || []).map((f) => ({ name: f.name, path: f.path, size: f.size }));
  // Whenever base64 is available (paste, picker, native-drop preview) the image is uploaded: a raw disk
  // path would be rejected by /upload/raw's whitelist and the thumbnail would break after send.
  const imageMetas: { name: string; path: string }[] = [];
  for (const img of images || []) {
    const dataUrl = img.base64 || (img.path?.startsWith('data:') ? img.path : undefined);
    if (dataUrl) {
      imageMetas.push({ name: img.name, path: await uploadDataUrl(sessionId, img.name, dataUrl) });
    } else if (img.path && img.path !== img.name) {
      imageMetas.push({ name: img.name, path: img.path });
    }
  }
  const data = await postJson(`/session/${sessionId}/prompt`, { sessionId, prompt, display: prompt, files: filesMeta, imageMetas });
  return data.userMessageId;
}

export async function pollMessages(sessionId: string, afterId?: string, limit = 50): Promise<PollResult> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (afterId) params.set('after', afterId);
  const data = await getJson(`/session/${sessionId}/messages?${params}`);
  return {
    messages: (data.messages || []).map((m: Record<string, unknown>) => normalizeMessage(m)),
    partial: data.partial ? normalizeMessage(data.partial, 'in_progress') : undefined,
    status: data.status,
    plan: data.plan,
    model: data.model,
  };
}

export const cancelGeneration = (sessionId: string) => postJson(`/session/${sessionId}/cancel`, { sessionId }).then(() => undefined);

export function setSessionModel(sessionId: string, llmNo: number): Promise<{ ok: boolean; llmNo: number; model: LiveModel }> {
  return postJson(`/session/${sessionId}/model`, { llmNo }) as Promise<{ ok: boolean; llmNo: number; model: LiveModel }>;
}

/** Conductor worker sessions (`tui_` prefix) are internal dispatch and never listed. */
export async function listSessions(): Promise<SessionInfo[]> {
  const data = await getJson<{ sessions?: SessionInfo[] }>('/sessions');
  return (data.sessions || []).filter((s) => !s.id.startsWith('tui_'));
}

export const deleteSession = (sessionId: string) => deleteJson(`/session/${sessionId}`).then(() => undefined);
export const renameSession = (sessionId: string, title: string) => patchJson(`/session/${sessionId}`, { title }).then(() => undefined);
export const pinSession = (sessionId: string, pinned: boolean) => patchJson(`/session/${sessionId}`, { pinned }).then(() => undefined);
