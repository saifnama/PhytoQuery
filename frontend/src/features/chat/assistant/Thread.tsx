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

import { type FC } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useMessage,
  useThread,
} from '@assistant-ui/react';
import { ArrowUp, DownloadSimple } from '@phosphor-icons/react';
import type { RagMessageCustomData, RagSource } from './runtime';
import { exportAnswerAsPdf } from './exportPdf';

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
      </ThreadPrimitive.Viewport>

      <Composer />
    </ThreadPrimitive.Root>
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
    <div className="max-w-[80%] rounded-2xl bg-blue-600 px-4 py-2 text-white shadow-sm">
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
  const thread = useThread();
  const customData = (message.metadata?.custom ?? {}) as RagMessageCustomData;
  const sources = customData.sources ?? [];

  const handleExportPdf = () => {
    // Find the user message immediately preceding this assistant turn
    // (skipping any system messages) so the PDF carries the question.
    const ourIndex = thread.messages.findIndex((m) => m.id === message.id);
    let priorUserText = '';
    if (ourIndex > 0) {
      for (let i = ourIndex - 1; i >= 0; i--) {
        const candidate = thread.messages[i];
        if (candidate.role === 'user') {
          priorUserText = readMessageText(candidate);
          break;
        }
      }
    }

    exportAnswerAsPdf({
      question: priorUserText,
      answer: readMessageText(message),
      sources,
      timestamp: message.createdAt
        ? new Date(message.createdAt).toISOString()
        : undefined,
    });
  };

  return (
    <MessagePrimitive.Root className="flex justify-start">
      <div className="max-w-[85%] space-y-3">
        <div className="rounded-2xl bg-slate-100 px-4 py-3 text-slate-900 shadow-sm">
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

        <div className="flex items-center gap-2 pt-1">
          <button
            type="button"
            onClick={handleExportPdf}
            className="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50 hover:text-slate-900"
            title="Download this answer as a PDF"
          >
            <DownloadSimple size={14} weight="bold" />
            <span>Export PDF</span>
          </button>
        </div>
      </div>
    </MessagePrimitive.Root>
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
  <ComposerPrimitive.Root className="border-t border-slate-200 bg-white p-3">
    <div className="flex items-end gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 focus-within:border-blue-400 focus-within:bg-white">
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="Ask about your papers…"
        className="flex-1 resize-none bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
      />
      <ComposerPrimitive.Send asChild>
        <button
          type="submit"
          className="rounded-lg bg-blue-600 p-2 text-white transition-colors hover:bg-blue-700 disabled:opacity-40"
          aria-label="Send message"
        >
          <ArrowUp size={16} weight="bold" />
        </button>
      </ComposerPrimitive.Send>
    </div>
  </ComposerPrimitive.Root>
);
