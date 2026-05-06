/**
 * assistant-ui chat thread for the PhytoQuery RAG page.
 *
 * Composes ThreadPrimitive + MessagePrimitive + ComposerPrimitive into
 * a single <Thread /> component. Custom slots:
 *   - AssistantMessage renders text via react-markdown and follows it
 *     with a source-pills row, fed from message.metadata.custom.sources
 *     (set by the runtime adapter in ./runtime.ts).
 *   - UserMessage stays plain text inside a styled bubble.
 *
 * Tailwind-only styling so it composes cleanly with the rest of the
 * app (and future daisyUI adoption).
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
    <ThreadPrimitive.Root className="flex h-full flex-col bg-white">
      <ThreadHeader />
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        <ThreadPrimitive.Empty>
          <div className="flex h-full items-center justify-center text-slate-500">
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
    <div className="flex items-center justify-end border-b border-slate-100 bg-white/80 px-6 py-2 backdrop-blur-sm">
      <button
        type="button"
        onClick={handleExportChat}
        className="flex items-center gap-1.5 rounded-lg border border-transparent px-3 py-1.5 text-xs font-medium text-slate-500 transition-colors hover:border-slate-200 hover:bg-slate-100 hover:text-slate-800"
        title="Download the entire chat as a PDF"
      >
        <ArrowDown size={14} weight="bold" />
        <span>Export Chat (PDF)</span>
      </button>
    </div>
  );
};

/** Three-dot bouncing indicator shown only while the assistant is
 * still streaming/computing a response. Mirrors the loading dots from
 * the pre-migration UI. */
const TypingIndicator: FC = () => {
  const thread = useThread();
  if (!thread.isRunning) return null;
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl border border-slate-100 bg-white px-5 py-3 shadow-sm">
        <div className="flex items-center space-x-1.5">
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300" />
          <span
            className="h-2 w-2 animate-bounce rounded-full bg-slate-300"
            style={{ animationDelay: '0.1s' }}
          />
          <span
            className="h-2 w-2 animate-bounce rounded-full bg-slate-300"
            style={{ animationDelay: '0.2s' }}
          />
        </div>
      </div>
    </div>
  );
};

const DefaultEmpty: FC = () => (
  <div className="max-w-md text-center">
    <h2 className="text-xl font-semibold text-slate-900">Ask about your papers</h2>
    <p className="mt-2 text-sm text-slate-600">
      Upload PDFs in the sidebar, then ask questions. Answers cite the
      source paper and section.
    </p>
  </div>
);

const UserMessage: FC = () => (
  <MessagePrimitive.Root className="flex justify-end">
    <div
      className="max-w-[80%] rounded-2xl px-4 py-2 text-slate-900 shadow-sm"
      style={{ backgroundColor: PINK_USER_BG }}
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

/** Extract the plain-text body of a ThreadMessage (concatenates all
 * text parts, ignores tool/file parts that don't apply here). */
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
    <MessagePrimitive.Root className="flex justify-start">
      <div className="max-w-[85%] space-y-3">
        <div className="rounded-2xl border border-slate-100 bg-white px-5 py-3 text-slate-900 shadow-sm">
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
          <SourcePills sources={sources} onSourceClick={onSourceClick} />
        )}

        <CopyMessageButton text={readMessageText(message)} />
      </div>
    </MessagePrimitive.Root>
  );
};

/** Per-message Copy button. Restores the copy-to-clipboard affordance
 * the pre-migration UI had on each assistant bubble. */
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
    <div className="flex items-center gap-2 pt-0.5">
      <button
        type="button"
        onClick={handleCopy}
        className="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-500 transition-colors hover:bg-slate-50 hover:text-slate-800"
        title={copied ? 'Copied!' : 'Copy answer'}
        aria-label={copied ? 'Copied!' : 'Copy answer'}
      >
        {copied ? (
          <>
            <Check size={14} weight="bold" className="text-green-600" />
            <span>Copied</span>
          </>
        ) : (
          <>
            <CopySimple size={14} weight="regular" />
            <span>Copy</span>
          </>
        )}
      </button>
    </div>
  );
};

interface SourcePillsProps {
  sources: RagSource[];
  onSourceClick?: (source: RagSource) => void;
}

const SourcePills: FC<SourcePillsProps> = ({ sources, onSourceClick }) => (
  <div className="flex flex-wrap gap-2">
    {sources.map((source, idx) => {
      const label = source.section
        ? `${source.source} · ${source.section}`
        : source.source;
      const scoreColor =
        source.score >= 80
          ? 'border-green-300 bg-green-50 text-green-700'
          : source.score >= 60
            ? 'border-yellow-300 bg-yellow-50 text-yellow-700'
            : 'border-slate-300 bg-slate-50 text-slate-600';
      return (
        <button
          key={`${source.source}-${idx}`}
          type="button"
          onClick={() => onSourceClick?.(source)}
          className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors hover:bg-white ${scoreColor}`}
          title={source.chunk_text.slice(0, 240)}
        >
          <span className="truncate max-w-[200px]">{label}</span>
          <span className="text-[10px] opacity-70">{source.score}%</span>
        </button>
      );
    })}
  </div>
);

const Composer: FC = () => (
  <ComposerPrimitive.Root className="border-t border-slate-200 bg-white p-4">
    <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-[28px] border border-slate-200/80 bg-white px-2 py-1.5 shadow-2xl shadow-slate-200/50 focus-within:border-slate-300">
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="Ask anything..."
        className="min-h-[44px] flex-1 resize-none bg-transparent px-3 py-2.5 text-base text-slate-800 placeholder:text-slate-400 focus:outline-none"
      />
      <div className="pb-1.5">
        <ComposerPrimitive.Send asChild>
          <button
            type="submit"
            className="group flex h-10 w-10 items-center justify-center rounded-full text-white shadow-md transition-all hover:shadow-lg active:scale-95 disabled:opacity-40 disabled:shadow-none"
            style={{ backgroundColor: PINK_ACCENT }}
            aria-label="Send message"
          >
            <ArrowUp
              size={18}
              weight="bold"
              className="transition-transform group-hover:-translate-y-px"
            />
          </button>
        </ComposerPrimitive.Send>
      </div>
    </div>
  </ComposerPrimitive.Root>
);
