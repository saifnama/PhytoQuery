/**
 * assistant-ui chat thread for the PhytoQuery RAG page — daisyUI build.
 *
 * Composes ThreadPrimitive + MessagePrimitive + ComposerPrimitive into
 * a single <Thread /> component, with all surface styling expressed
 * through daisyUI component classes (`chat chat-end chat-bubble`,
 * `btn btn-primary btn-circle`, `loading loading-dots`, `textarea`,
 * etc.) so the component picks up your daisyUI theme tokens
 * automatically.
 *
 * Custom slots:
 *   - AssistantMessage renders text via react-markdown and follows it
 *     with a source-pills row, fed from message.metadata.custom.sources.
 *   - UserMessage uses the chat-bubble pattern with the pink brand
 *     color preserved via inline style (so daisyUI theme primary
 *     doesn't shadow the existing branding).
 */

import { type FC, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useMessage,
  useThread,
} from '@assistant-ui/react';
import { ArrowUp, ArrowDown, CopySimple, Check } from '@phosphor-icons/react';
import type { RagMessageCustomData, RagSource } from './runtime';
import { exportThreadAsPdf, type ThreadTurn } from './exportPdf';

const PINK_ACCENT = '#ff6dba';
const PINK_USER_BG = '#ffecf6';

interface ThreadProps {
  /** Optional: invoked when the user clicks a source pill. The parent
   * (RagPage) handles opening the PDF preview modal. */
  onSourceClick?: (source: RagSource) => void;
  /** Welcome card content shown when the thread is empty. */
  emptyContent?: React.ReactNode;
}

export const Thread: FC<ThreadProps> = ({ onSourceClick, emptyContent }) => {
  return (
    <ThreadPrimitive.Root className="flex h-full flex-col bg-base-100">
      <ThreadHeader />
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto px-4 py-6">
        <ThreadPrimitive.Empty>
          <div className="flex h-full items-center justify-center text-base-content/60">
            {emptyContent ?? <DefaultEmpty />}
          </div>
        </ThreadPrimitive.Empty>

        <ThreadPrimitive.Messages
          components={{
            UserMessage: UserMessage,
            AssistantMessage: () => <AssistantMessage onSourceClick={onSourceClick} />,
          }}
        />

        <TypingIndicator />
      </ThreadPrimitive.Viewport>

      <Composer />
    </ThreadPrimitive.Root>
  );
};

/** Top bar with the whole-chat Export PDF button. Hidden when there
 * are no messages so it doesn't clutter the empty state. */
const ThreadHeader: FC = () => {
  const thread = useThread();
  if (thread.messages.length === 0) return null;

  const handleExportChat = () => {
    const turns: ThreadTurn[] = thread.messages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => {
        const text = readMessageText(m);
        const role: 'user' | 'assistant' =
          m.role === 'user' ? 'user' : 'assistant';
        const customData = (m.metadata?.custom ?? {}) as RagMessageCustomData;
        return {
          role,
          text,
          sources: role === 'assistant' ? customData.sources : undefined,
        };
      });
    exportThreadAsPdf({ turns });
  };

  return (
    <div className="flex items-center justify-end border-b border-base-200 bg-base-100/80 px-6 py-2 backdrop-blur-sm">
      <button
        type="button"
        onClick={handleExportChat}
        className="btn btn-ghost btn-sm gap-1.5"
        title="Download the entire chat as a PDF"
      >
        <ArrowDown size={14} weight="bold" />
        <span>Export Chat (PDF)</span>
      </button>
    </div>
  );
};

/** daisyUI loading-dots indicator shown while the assistant is still
 * streaming/computing a response. Wrapped in a chat-bubble so it sits
 * in the message column at the start side. */
const TypingIndicator: FC = () => {
  const thread = useThread();
  if (!thread.isRunning) return null;
  return (
    <div className="chat chat-start">
      <div className="chat-bubble chat-bubble-neutral bg-base-200 text-base-content">
        <span className="loading loading-dots loading-md" />
      </div>
    </div>
  );
};

const DefaultEmpty: FC = () => (
  <div className="card max-w-md bg-base-100 shadow-none">
    <div className="card-body items-center text-center">
      <h2 className="card-title text-base-content">Ask about your papers</h2>
      <p className="text-sm text-base-content/70">
        Upload PDFs in the sidebar, then ask questions. Answers cite the
        source paper and section.
      </p>
    </div>
  </div>
);

const UserMessage: FC = () => (
  <MessagePrimitive.Root className="chat chat-end">
    <div
      className="chat-bubble shadow-sm"
      style={{ backgroundColor: PINK_USER_BG, color: '#1f2937' }}
    >
      <MessagePrimitive.Content
        components={{
          Text: ({ text }) => <span className="whitespace-pre-wrap">{text}</span>,
        }}
      />
    </div>
  </MessagePrimitive.Root>
);

interface AssistantMessageProps {
  onSourceClick?: (source: RagSource) => void;
}

/** Extract the plain-text body of a ThreadMessage. */
function readMessageText(message: { content: readonly { type: string; text?: string }[] | undefined }): string {
  if (!message?.content) return '';
  return message.content
    .map((part) => (part.type === 'text' ? part.text ?? '' : ''))
    .join('')
    .trim();
}

const AssistantMessage: FC<AssistantMessageProps> = ({ onSourceClick }) => {
  const message = useMessage();
  const customData = (message.metadata?.custom ?? {}) as RagMessageCustomData;
  const sources = customData.sources ?? [];

  return (
    <MessagePrimitive.Root className="chat chat-start">
      <div className="chat-bubble bg-base-100 border border-base-200 text-base-content shadow-sm">
        <MessagePrimitive.Content
          components={{
            Text: ({ text }) => (
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{text}</ReactMarkdown>
              </div>
            ),
          }}
        />
      </div>

      {sources.length > 0 && (
        <div className="chat-footer mt-2">
          <SourcePills sources={sources} onSourceClick={onSourceClick} />
        </div>
      )}

      <div className="chat-footer mt-2">
        <CopyMessageButton text={readMessageText(message)} />
      </div>
    </MessagePrimitive.Root>
  );
};

/** Per-message Copy button using daisyUI btn classes. */
const CopyMessageButton: FC<{ text: string }> = ({ text }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.error('Copy failed:', err);
    }
  };
  return (
    <button
      type="button"
      onClick={handleCopy}
      className="btn btn-ghost btn-xs gap-1"
      title={copied ? 'Copied!' : 'Copy answer'}
      aria-label={copied ? 'Copied!' : 'Copy answer'}
    >
      {copied ? (
        <>
          <Check size={14} weight="bold" className="text-success" />
          <span>Copied</span>
        </>
      ) : (
        <>
          <CopySimple size={14} weight="regular" />
          <span>Copy</span>
        </>
      )}
    </button>
  );
};

interface SourcePillsProps {
  sources: RagSource[];
  onSourceClick?: (source: RagSource) => void;
}

const SourcePills: FC<SourcePillsProps> = ({ sources, onSourceClick }) => (
  <div className="flex flex-wrap gap-1.5">
    {sources.map((source, idx) => {
      const label = source.section
        ? `${source.source} · ${source.section}`
        : source.source;
      // Map score to a daisyUI semantic badge color.
      const badgeClass =
        source.score >= 80
          ? 'badge-success'
          : source.score >= 60
            ? 'badge-warning'
            : 'badge-ghost';
      return (
        <button
          key={`${source.source}-${idx}`}
          type="button"
          onClick={() => onSourceClick?.(source)}
          className={`badge badge-sm ${badgeClass} gap-1 cursor-pointer hover:badge-outline`}
          title={source.chunk_text.slice(0, 240)}
        >
          <span className="truncate max-w-[200px]">{label}</span>
          <span className="opacity-70">{source.score}%</span>
        </button>
      );
    })}
  </div>
);

const Composer: FC = () => (
  <ComposerPrimitive.Root className="border-t border-base-200 bg-base-100 p-4">
    <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-[28px] border border-base-200 bg-base-100 px-2 py-1.5 shadow-2xl shadow-base-300/40 focus-within:border-base-300">
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="Ask anything..."
        className="textarea textarea-ghost min-h-[44px] flex-1 resize-none bg-transparent text-base focus:outline-none focus:bg-transparent"
      />
      <div className="pb-1">
        <ComposerPrimitive.Send asChild>
          <button
            type="submit"
            className="btn btn-circle btn-md border-none text-white shadow-md transition-all hover:shadow-lg active:scale-95 disabled:opacity-40 disabled:shadow-none"
            style={{ backgroundColor: PINK_ACCENT }}
            aria-label="Send message"
          >
            <ArrowUp size={18} weight="bold" />
          </button>
        </ComposerPrimitive.Send>
      </div>
    </div>
  </ComposerPrimitive.Root>
);
