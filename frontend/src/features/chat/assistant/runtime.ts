/**
 * assistant-ui ChatModelAdapter for the PhytoQuery RAG backend.
 *
 * Bridges assistant-ui's thread state to our existing
 * `/api/chat/query/json` endpoint:
 *   - Maps assistant-ui ThreadMessage[] -> our { role, content } chat_history
 *   - Sends `selected_files` derived from current React state via a
 *     getter callback (so the adapter never captures a stale selection)
 *   - Calls `fetch` directly (not `ragApi.query`) so we can pass the
 *     `AbortSignal` assistant-ui hands us — `ragApi.query` doesn't
 *     accept one and we don't want to widen its signature in this PR
 *   - Returns the answer as a single text part (no streaming yet —
 *     backend currently returns the full response in one shot)
 *   - Stashes returned sources on the message metadata so a custom
 *     React slot can render the source-pill row under the answer
 *
 * The hook returns a runtime ready to be passed into
 * <AssistantRuntimeProvider runtime={runtime}>.
 */

import type {
  ChatModelAdapter,
  ChatModelRunResult,
  ThreadHistoryAdapter,
  ThreadMessage,
  ThreadMessageLike,
} from '@assistant-ui/react';
import { useLocalRuntime } from '@assistant-ui/react';
import { useMemo, useRef } from 'react';
import { historyKeyFor } from './threadStore';

/** Legacy single-thread key. Kept so a user mid-session who hasn't
 * gotten their thread list bootstrapped yet still sees their existing
 * conversation. New code paths use historyKeyFor(threadId) instead. */
const LEGACY_HISTORY_STORAGE_KEY = 'pq_chat_history';

/** Source attribution as our backend returns it. */
export interface RagSource {
  source: string;
  section: string;
  parser_type: string;
  score: number;
  chunk_text: string;
}

/** Custom metadata key our messages carry so the UI can render sources. */
export interface RagMessageCustomData {
  sources?: RagSource[];
}

function extractMessageText(message: ThreadMessage): string {
  if (!message.content) return '';
  return message.content
    .map((part) => (part.type === 'text' ? part.text : ''))
    .join('')
    .trim();
}

function toBackendChatHistory(
  messages: readonly ThreadMessage[],
): Array<{ role: 'user' | 'assistant'; content: string }> {
  // Drop the trailing user message — our backend takes the current
  // question separately as `query`, with prior turns as chat_history.
  // assistant-ui passes ALL messages including the new user one, so we
  // strip the final user turn before sending.
  const trimmed = [...messages];
  if (trimmed.length > 0 && trimmed[trimmed.length - 1].role === 'user') {
    trimmed.pop();
  }
  return trimmed
    .map((m) => {
      const role: 'user' | 'assistant' = m.role === 'user' ? 'user' : 'assistant';
      const content = extractMessageText(m);
      return content ? { role, content } : null;
    })
    .filter((entry): entry is { role: 'user' | 'assistant'; content: string } => entry !== null);
}

interface QueryResponseShape {
  answer?: string;
  sources?: RagSource[];
}

async function postQuery(
  question: string,
  selectedFiles: string[],
  chatHistory: Array<{ role: 'user' | 'assistant'; content: string }>,
  signal: AbortSignal,
): Promise<QueryResponseShape> {
  const response = await fetch('/api/chat/query/json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: question,
      selected_files: selectedFiles.length > 0 ? selectedFiles : undefined,
      chat_history: chatHistory.length > 0 ? chatHistory : undefined,
    }),
    credentials: 'include',
    signal,
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(`Query failed (${response.status}): ${detail}`);
  }
  return response.json();
}

/** sessionStorage-backed history adapter for a SPECIFIC thread.
 *
 * - When ``threadId`` is provided, history is stored at
 *   ``pq_thread_<threadId>_history`` and is per-thread.
 * - When ``threadId`` is undefined (no thread list initialized yet),
 *   falls back to the legacy single-thread key ``pq_chat_history`` so
 *   pre-existing conversations don't disappear on first load after
 *   the multi-thread upgrade.
 */
function createSessionHistoryAdapter(threadId: string | undefined): ThreadHistoryAdapter {
  const key = threadId ? historyKeyFor(threadId) : LEGACY_HISTORY_STORAGE_KEY;
  return {
    async load() {
      try {
        const raw = sessionStorage.getItem(key);
        if (!raw) return { messages: [] };
        const parsed = JSON.parse(raw);
        if (parsed && Array.isArray(parsed.messages)) {
          return parsed;
        }
        return { messages: [] };
      } catch {
        return { messages: [] };
      }
    },
    async append(item) {
      try {
        const raw = sessionStorage.getItem(key);
        const repo =
          raw && JSON.parse(raw).messages
            ? JSON.parse(raw)
            : { messages: [] as Array<unknown> };
        repo.messages.push(item);
        sessionStorage.setItem(key, JSON.stringify(repo));
      } catch (e) {
        console.warn('Failed to persist chat history:', e);
      }
    },
  };
}

export interface PhytoQueryRuntimeOptions {
  /** Returns the list of currently-selected source filenames. Called on
   * every send so the user's checkbox state always reflects in the
   * outgoing request without re-creating the adapter. */
  getSelectedFiles: () => string[];
  /** Initial messages to seed the thread with on first mount. Used for
   * "import paper from /search" handoff that pre-populates a question
   * about the imported PDF. */
  initialMessages?: readonly ThreadMessageLike[];
  /** When true, persists thread state to sessionStorage so it survives
   * a page reload within the same browser session. Default true. */
  enableSessionPersistence?: boolean;
  /** Active thread id. When provided, history is keyed per-thread so
   * users can maintain multiple parallel chats. Falls back to a single
   * legacy key when undefined. */
  threadId?: string;
}

export function usePhytoQueryRuntime(opts: PhytoQueryRuntimeOptions) {
  // Stash the latest getter in a ref so the adapter always reads the
  // current React state without being recreated on every render.
  const getSelectedFilesRef = useRef(opts.getSelectedFiles);
  getSelectedFilesRef.current = opts.getSelectedFiles;

  const adapter = useMemo<ChatModelAdapter>(
    () => ({
      async run({ messages, abortSignal }): Promise<ChatModelRunResult> {
        const lastUser = [...messages].reverse().find((m) => m.role === 'user');
        const question = lastUser ? extractMessageText(lastUser) : '';

        if (!question) {
          return { content: [{ type: 'text', text: '' }] };
        }

        const chatHistory = toBackendChatHistory(messages);
        const selectedFiles = getSelectedFilesRef.current();

        const result = await postQuery(
          question,
          selectedFiles,
          chatHistory,
          abortSignal,
        );

        const sources = result.sources ?? [];

        return {
          content: [{ type: 'text', text: result.answer ?? '' }],
          metadata: {
            // Attach sources as custom metadata so a UI slot can render
            // a source-pill row under the assistant's reply.
            custom: { sources } satisfies RagMessageCustomData,
          },
        };
      },
    }),
    [],
  );

  const history = useMemo<ThreadHistoryAdapter | undefined>(
    () =>
      opts.enableSessionPersistence === false
        ? undefined
        : createSessionHistoryAdapter(opts.threadId),
    [opts.enableSessionPersistence, opts.threadId],
  );

  return useLocalRuntime(adapter, {
    initialMessages: opts.initialMessages,
    adapters: history ? { history } : undefined,
  });
}

/** Public so the parent (RagPage's "Reset all" handler) can clear the
 * legacy single-thread history. Multi-thread state is wiped via
 * ``clearAllThreads`` from ./threadStore. */
export function clearPersistedChatHistory(): void {
  try {
    sessionStorage.removeItem(LEGACY_HISTORY_STORAGE_KEY);
  } catch {
    /* noop */
  }
}
