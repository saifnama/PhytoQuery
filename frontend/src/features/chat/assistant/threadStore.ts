/**
 * sessionStorage-backed multi-thread store.
 *
 * Thread metadata (id, title, createdAt) lives at the single key
 * ``pq_thread_list``. Each thread's message history lives at
 * ``pq_thread_<id>_history`` and is read/written by the runtime
 * adapter (see ./runtime.ts).
 *
 * No backend; threads survive page reloads inside the same browser
 * session and are cleared when the tab closes (sessionStorage rules).
 */

const THREAD_LIST_KEY = 'pq_thread_list';
const THREAD_HISTORY_PREFIX = 'pq_thread_';
const THREAD_HISTORY_SUFFIX = '_history';

export interface StoredThread {
  id: string;
  title: string;
  createdAt: string;
}

interface StoredThreadListShape {
  threads: StoredThread[];
}

function safeParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
}

export function loadThreads(): StoredThread[] {
  if (typeof window === 'undefined') return [];
  const raw = sessionStorage.getItem(THREAD_LIST_KEY);
  const parsed = safeParse<StoredThreadListShape>(raw, { threads: [] });
  return Array.isArray(parsed.threads) ? parsed.threads : [];
}

function persistThreads(threads: StoredThread[]): void {
  try {
    sessionStorage.setItem(
      THREAD_LIST_KEY,
      JSON.stringify({ threads } satisfies StoredThreadListShape),
    );
  } catch (e) {
    console.warn('Failed to persist thread list:', e);
  }
}

export function createThread(title?: string): StoredThread {
  const thread: StoredThread = {
    id:
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `t_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`,
    title: title ?? 'New chat',
    createdAt: new Date().toISOString(),
  };
  const threads = loadThreads();
  threads.unshift(thread);
  persistThreads(threads);
  return thread;
}

export function deleteThread(id: string): void {
  const threads = loadThreads().filter((t) => t.id !== id);
  persistThreads(threads);
  try {
    sessionStorage.removeItem(historyKeyFor(id));
  } catch {
    /* noop */
  }
}

export function renameThread(id: string, title: string): void {
  const threads = loadThreads().map((t) =>
    t.id === id ? { ...t, title } : t,
  );
  persistThreads(threads);
}

/** Per-thread history sessionStorage key. */
export function historyKeyFor(threadId: string): string {
  return `${THREAD_HISTORY_PREFIX}${threadId}${THREAD_HISTORY_SUFFIX}`;
}

/**
 * Derive a thread title from its current persisted history. Returns the
 * existing title if the thread has no messages yet (no first user
 * message to base a title on).
 */
export function deriveTitleFromHistory(
  threadId: string,
  fallback: string,
): string {
  try {
    const raw = sessionStorage.getItem(historyKeyFor(threadId));
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    const messages: Array<{
      message?: { role?: string; content?: Array<{ type?: string; text?: string }> };
    }> = parsed?.messages ?? [];
    const firstUser = messages.find((m) => m.message?.role === 'user');
    if (!firstUser?.message?.content) return fallback;
    const text = firstUser.message.content
      .map((part) => (part.type === 'text' ? part.text ?? '' : ''))
      .join('')
      .trim();
    if (!text) return fallback;
    return text.length > 60 ? `${text.slice(0, 57)}…` : text;
  } catch {
    return fallback;
  }
}

/** Clear the entire multi-thread state (used by Reset all). */
export function clearAllThreads(): void {
  try {
    const threads = loadThreads();
    threads.forEach((t) => sessionStorage.removeItem(historyKeyFor(t.id)));
    sessionStorage.removeItem(THREAD_LIST_KEY);
    // Also remove the legacy single-thread key.
    sessionStorage.removeItem('pq_chat_history');
  } catch (e) {
    console.warn('Failed to clear thread store:', e);
  }
}
