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
  ThreadMessage,
} from '@assistant-ui/react';
import { useLocalRuntime } from '@assistant-ui/react';
import { useMemo, useRef } from 'react';

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

export interface PhytoQueryRuntimeOptions {
  /** Returns the list of currently-selected source filenames. Called on
   * every send so the user's checkbox state always reflects in the
   * outgoing request without re-creating the adapter. */
  getSelectedFiles: () => string[];
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

  return useLocalRuntime(adapter);
}
