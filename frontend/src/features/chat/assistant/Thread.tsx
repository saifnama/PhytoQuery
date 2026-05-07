/**
 * assistant-ui chat thread for the PhytoQuery RAG page — daisyUI build.
 *
 * Composes ThreadPrimitive + MessagePrimitive + ComposerPrimitive +
 * ActionBarPrimitive + BranchPickerPrimitive into a single <Thread />
 * component, with all surface styling expressed through daisyUI
 * classes so the component picks up the theme automatically.
 *
 * Custom slots:
 *   - AssistantMessage renders text via MarkdownTextPrimitive (smooth
 *     streaming + memoized markdown components) and follows it with a
 *     source-pills row from message.metadata.custom.sources, plus an
 *     ActionBar (Copy / Reload / Speak) and a BranchPicker.
 *   - UserMessage uses the chat-bubble pattern with the pink brand
 *     color preserved as inline style. It renders an inline Composer
 *     when in edit mode so users can revise prior questions.
 *   - The Composer swaps between Send and Cancel via ThreadPrimitive.If
 *     so users can stop slow LLM calls.
 *   - DefaultEmpty includes ThreadPrimitive.Suggestion starter prompts
 *     so first-time users have one-click ways to begin.
 */

import { type FC } from 'react';
import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useMessage,
  useThread,
} from '@assistant-ui/react';
import {
  MarkdownTextPrimitive,
  unstable_memoizeMarkdownComponents as memoizeMarkdownComponents,
} from '@assistant-ui/react-markdown';
import {
  ArrowUp,
  ArrowDown,
  CopySimple,
  Check,
  Stop,
  ArrowClockwise,
  PencilSimple,
  SpeakerHigh,
  SpeakerSlash,
  CaretLeft,
  CaretRight,
  X,
  FilePdf,
} from '@phosphor-icons/react';
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
      <ThreadPrimitive.Viewport className="relative flex-1 overflow-y-auto px-4 py-6">
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
        <ScrollToBottomButton />
      </ThreadPrimitive.Viewport>

      <Composer />
    </ThreadPrimitive.Root>
  );
};

/** Whole-chat PDF export button — sits inside the assistant
 * ActionBar so users can export the conversation without scrolling
 * to a header. Each assistant turn renders one of these; clicking
 * any of them produces the same full-chat PDF. */
const ExportChatPdfButton: FC = () => {
  const thread = useThread();

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
    <button
      type="button"
      onClick={handleExportChat}
      className="btn btn-ghost btn-xs btn-square"
      title="Export entire chat as PDF"
      aria-label="Export entire chat as PDF"
    >
      <FilePdf size={14} weight="regular" />
    </button>
  );
};

/** daisyUI loading-dots indicator shown while the assistant is still
 * streaming/computing a response. */
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

/** Floating "scroll to latest" button. Auto-disabled when already at
 * the bottom — clicking it smooth-scrolls the viewport. */
const ScrollToBottomButton: FC = () => (
  <ThreadPrimitive.ScrollToBottom asChild>
    <button
      type="button"
      className="btn btn-circle btn-sm absolute bottom-4 right-4 shadow-md disabled:hidden"
      aria-label="Scroll to latest message"
      title="Scroll to latest"
    >
      <ArrowDown size={16} weight="bold" />
    </button>
  </ThreadPrimitive.ScrollToBottom>
);

/** Empty state — a quiet welcome card. No starter prompts; users
 * type their own question in the composer. */
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

/** Memoized markdown components — react-markdown re-renders every node
 * on each text update; memoizing each tag means only changed nodes
 * re-render. Big win once streaming lands. */
const markdownComponents = memoizeMarkdownComponents({});

/** Markdown renderer — uses MarkdownTextPrimitive with smooth
 * character-by-character render so streamed answers paint
 * progressively. Lives in a `prose` container for daisyUI typography. */
const MarkdownText: FC = () => (
  <MarkdownTextPrimitive
    smooth
    components={markdownComponents}
    className="prose prose-sm max-w-none"
  />
);

const UserMessage: FC = () => (
  <MessagePrimitive.Root className="chat chat-end group">
    {/* Edit-mode inline composer. When the user clicks ActionBar.Edit
     * on a previous question, the message swaps to a textarea so they
     * can revise — submitting forks a new branch from this point. */}
    <ComposerPrimitive.If editing>
      <UserEditComposer />
    </ComposerPrimitive.If>

    <ComposerPrimitive.If editing={false}>
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

      <UserActionBar />

      <BranchPicker />
    </ComposerPrimitive.If>
  </MessagePrimitive.Root>
);

/** Inline composer rendered inside a user message when it's in edit
 * mode. Submitting forks a new branch + re-runs the assistant. */
const UserEditComposer: FC = () => (
  <ComposerPrimitive.Root className="w-full max-w-2xl">
    <div className="flex flex-col gap-2 rounded-2xl border border-base-300 bg-base-100 p-2 shadow-sm">
      <ComposerPrimitive.Input
        className="textarea textarea-ghost min-h-[60px] resize-none bg-transparent text-sm focus:outline-none"
        autoFocus
      />
      <div className="flex justify-end gap-2">
        <ComposerPrimitive.Cancel asChild>
          <button type="button" className="btn btn-ghost btn-sm gap-1">
            <X size={14} weight="bold" />
            Cancel
          </button>
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send asChild>
          <button
            type="submit"
            className="btn btn-sm gap-1 border-none text-white"
            style={{ backgroundColor: PINK_ACCENT }}
          >
            <Check size={14} weight="bold" />
            Update
          </button>
        </ComposerPrimitive.Send>
      </div>
    </div>
  </ComposerPrimitive.Root>
);

/** Action bar for user messages — just Edit, hover-only. */
const UserActionBar: FC = () => (
  <ActionBarPrimitive.Root
    autohide="not-last"
    className="chat-footer mt-1 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 data-[floating=true]:opacity-100"
  >
    <ActionBarPrimitive.Edit asChild>
      <button
        type="button"
        className="btn btn-ghost btn-xs btn-square"
        title="Edit this question"
        aria-label="Edit question"
      >
        <PencilSimple size={12} weight="regular" />
      </button>
    </ActionBarPrimitive.Edit>
  </ActionBarPrimitive.Root>
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
    <MessagePrimitive.Root className="chat chat-start group">
      <div className="chat-bubble bg-base-100 border border-base-200 text-base-content shadow-sm">
        <MessagePrimitive.Content components={{ Text: MarkdownText }} />
      </div>

      {sources.length > 0 && (
        <div className="chat-footer mt-2">
          <SourcePills sources={sources} onSourceClick={onSourceClick} />
        </div>
      )}

      <AssistantActionBar />

      <BranchPicker />
    </MessagePrimitive.Root>
  );
};

/** Action bar for assistant messages — Copy, Reload (regenerate),
 * Speak (TTS via WebSpeechSynthesisAdapter wired in runtime.ts).
 * Hidden while the thread is running so it doesn't flicker. Visible
 * always on the last message; hover-only on older messages. */
const AssistantActionBar: FC = () => (
  <ActionBarPrimitive.Root
    hideWhenRunning
    autohide="not-last"
    autohideFloat="single-branch"
    className="chat-footer mt-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 data-[floating=true]:opacity-100 data-[autohide=never]:opacity-100"
  >
    <ActionBarPrimitive.Copy asChild>
      <button
        type="button"
        className="btn btn-ghost btn-xs btn-square"
        title="Copy answer"
        aria-label="Copy answer"
      >
        <MessagePrimitive.If copied>
          <Check size={14} weight="bold" className="text-success" />
        </MessagePrimitive.If>
        <MessagePrimitive.If copied={false}>
          <CopySimple size={14} weight="regular" />
        </MessagePrimitive.If>
      </button>
    </ActionBarPrimitive.Copy>

    <ActionBarPrimitive.Reload asChild>
      <button
        type="button"
        className="btn btn-ghost btn-xs btn-square"
        title="Regenerate this answer"
        aria-label="Regenerate"
      >
        <ArrowClockwise size={14} weight="regular" />
      </button>
    </ActionBarPrimitive.Reload>

    <ExportChatPdfButton />

    {/* Speak / StopSpeaking flip based on speaking state. */}
    <MessagePrimitive.If speaking={false}>
      <ActionBarPrimitive.Speak asChild>
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-square"
          title="Read this answer aloud"
          aria-label="Read aloud"
        >
          <SpeakerHigh size={14} weight="regular" />
        </button>
      </ActionBarPrimitive.Speak>
    </MessagePrimitive.If>
    <MessagePrimitive.If speaking>
      <ActionBarPrimitive.StopSpeaking asChild>
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-square"
          title="Stop reading"
          aria-label="Stop reading"
        >
          <SpeakerSlash size={14} weight="regular" />
        </button>
      </ActionBarPrimitive.StopSpeaking>
    </MessagePrimitive.If>
  </ActionBarPrimitive.Root>
);

/** Branch navigator — appears below a message that has alternative
 * branches (created when user edits or regenerates). Shows
 * "‹ N / Total ›" with arrow buttons for switching. */
const BranchPicker: FC = () => (
  <MessagePrimitive.If hasBranches>
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className="chat-footer mt-1 inline-flex items-center gap-0.5 text-xs text-base-content/60"
    >
      <BranchPickerPrimitive.Previous asChild>
        <button
          type="button"
          className="btn btn-ghost btn-xs px-1"
          aria-label="Previous branch"
          title="Previous branch"
        >
          <CaretLeft size={12} weight="bold" />
        </button>
      </BranchPickerPrimitive.Previous>
      <span className="tabular-nums px-1">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <button
          type="button"
          className="btn btn-ghost btn-xs px-1"
          aria-label="Next branch"
          title="Next branch"
        >
          <CaretRight size={12} weight="bold" />
        </button>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  </MessagePrimitive.If>
);

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
        {/* While the assistant is running, swap Send for a Cancel
         * button. ComposerPrimitive.Cancel calls the AbortSignal that
         * runtime.ts forwards to fetch — cancelling an in-flight LLM
         * call is immediate, no backend change needed. */}
        <ThreadPrimitive.If running>
          <ComposerPrimitive.Cancel asChild>
            <button
              type="button"
              className="btn btn-circle btn-md border-none bg-base-300 text-base-content shadow-md transition-all hover:bg-base-content/20 active:scale-95"
              aria-label="Stop generating"
              title="Stop"
            >
              <Stop size={18} weight="fill" />
            </button>
          </ComposerPrimitive.Cancel>
        </ThreadPrimitive.If>
        <ThreadPrimitive.If running={false}>
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
        </ThreadPrimitive.If>
      </div>
    </div>
  </ComposerPrimitive.Root>
);
