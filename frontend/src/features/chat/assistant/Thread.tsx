/**
 * assistant-ui chat thread for the PhytoQuery RAG page — daisyUI build.
 *
 * Citation rendering pipeline (industry-standard two-pass design):
 *   - Backend streams the answer with inline ``[<chunk_id>]`` markers
 *     (8-char hex IDs computed server-side per retrieved chunk).
 *   - ``MarkdownText`` runs a per-render ``preprocess`` that replaces
 *     each marker with ``[<sup>N</sup>](#cite-<chunk_id>)`` where N is
 *     a 1-based number assigned in order of first appearance — so the
 *     reader sees clean ``[1] [2]`` superscripts, while the chunk_id
 *     stays internal to the data layer.
 *   - The custom markdown ``a`` component (CitationLink) renders any
 *     ``#cite-…`` link as a clickable pink badge that calls back into
 *     RagPage to open the markdown-preview panel.
 *
 * No source-pill row anymore — superscripts are the only citation
 * affordance. Click a badge to view the cited chunk in the paper's
 * extracted markdown.
 */

import {
  type FC,
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useMemo,
} from 'react';
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
import type { Citation, RagMessageCustomData, RagSource } from './runtime';
import { exportThreadAsPdf, type ThreadTurn } from './exportPdf';

const PINK_ACCENT = '#ff6dba';
const PINK_USER_BG = '#ffecf6';

/** Payload delivered to RagPage when the user clicks a citation
 * superscript. ``source`` is set when the chunk_id resolves to a
 * known retrieved chunk; ``citation`` is set when Pass 2 produced a
 * verbatim quote for that chunk. Either may be undefined if the LLM
 * cited an id we no longer have (rare; we ignore those visually). */
export interface CitationClickPayload {
  chunkId: string;
  source?: RagSource;
  citation?: Citation;
}

interface ThreadProps {
  /** Invoked when the user clicks a citation superscript in an
   * assistant answer. Parent (RagPage) opens the markdown preview
   * panel and highlights the cited chunk + verbatim quote. */
  onCitationClick?: (payload: CitationClickPayload) => void;
  /** Welcome card content shown when the thread is empty. */
  emptyContent?: ReactNode;
}

/** Context that exposes the citation click handler to the static,
 * memoized markdown ``a`` override without re-creating the component
 * map per render. */
const CitationClickContext = createContext<
  ((chunkId: string) => void) | undefined
>(undefined);

export const Thread: FC<ThreadProps> = ({ onCitationClick, emptyContent }) => {
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
            AssistantMessage: () => (
              <AssistantMessage onCitationClick={onCitationClick} />
            ),
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
        Upload PDFs in the sidebar, then ask questions. Click a citation
        superscript to see the exact passage in the paper.
      </p>
    </div>
  </div>
);

/** Custom ``a`` component that intercepts ``#cite-<chunkId>`` links
 * (produced by ``MarkdownText``'s preprocess) and renders them as a
 * clickable pink badge. Click handler is read from
 * ``CitationClickContext`` so the memoized component map doesn't
 * need to be rebuilt per message. */
const CitationLink: FC<{
  href?: string;
  children?: ReactNode;
  className?: string;
}> = ({ href, children, ...rest }) => {
  const onCitationClick = useContext(CitationClickContext);
  if (typeof href === 'string' && href.startsWith('#cite-')) {
    const chunkId = href.slice('#cite-'.length);
    return (
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          onCitationClick?.(chunkId);
        }}
        className="citation-badge"
        title="View source passage"
        aria-label="View source for citation"
      >
        {children}
      </button>
    );
  }
  return (
    <a href={href} {...rest}>
      {children}
    </a>
  );
};

/** Memoized markdown components — react-markdown re-renders every
 * node on each text update; memoizing each tag means only changed
 * nodes re-render. Citation links are routed through CitationLink. */
const markdownComponents = memoizeMarkdownComponents({
  a: CitationLink,
});

/** Markdown renderer with citation preprocessing.
 *
 * The preprocess hook runs on every text update (smooth streaming)
 * and replaces ``[<chunk_id>]`` markers with markdown links of the
 * form ``[<sup>N</sup>](#cite-<chunk_id>)``. The numbering is built
 * deterministically from the order chunk_ids first appear in the
 * answer, so the same id always gets the same number for a given
 * answer text. Unknown ids (not in the message's ``sources``) are
 * left as-is so they don't crash — they just render as plain text. */
const MarkdownText: FC = () => {
  const message = useMessage();
  const customData = (message.metadata?.custom ?? {}) as RagMessageCustomData;
  const validChunkIds = useMemo(() => {
    const ids = new Set<string>();
    for (const s of customData.sources ?? []) {
      if (s.chunk_id) ids.add(s.chunk_id);
    }
    return ids;
  }, [customData.sources]);

  const preprocess = useCallback(
    (text: string) => {
      // Marker format is per-turn positional id. We accept both the
      // canonical ``[cN]`` and bare ``[N]`` because most LLMs drop
      // the ``c`` prefix — bare numeric brackets dominate their
      // training data and the LLM happily emits ``[1]`` even when
      // the prompt asks for ``[c1]``. Both forms normalize to
      // ``cN`` internally. ``validChunkIds`` bounds-checks against
      // the message's sources, so any bare ``[N]`` that doesn't map
      // to a real chunk (e.g. reference numbers from quoted paper
      // text) is left as plain prose — never rendered as a
      // clickable badge.
      const numbering = new Map<string, number>();
      let next = 1;
      // Whitespace inside the brackets is tolerated because real
      // LLMs emit padded forms like ``[ c1]`` or ``[ 1 ]`` in
      // practice (observed in production diagnostics). Without
      // this allowance, the marker stays as plain text and never
      // becomes a clickable badge.
      return text.replace(/\[\s*[Cc]?\s*(\d+)\s*\]/g, (match, num: string) => {
        const id = `c${num}`;
        if (!validChunkIds.has(id)) return match;
        if (!numbering.has(id)) {
          numbering.set(id, next);
          next += 1;
        }
        const n = numbering.get(id);
        return `[${n}](#cite-${id})`;
      });
    },
    [validChunkIds],
  );

  return (
    <MarkdownTextPrimitive
      smooth
      components={markdownComponents}
      preprocess={preprocess}
      className="prose prose-sm max-w-none"
    />
  );
};

const UserMessage: FC = () => (
  <MessagePrimitive.Root className="chat chat-end group">
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
  onCitationClick?: (payload: CitationClickPayload) => void;
}

/** Extract the plain-text body of a ThreadMessage. */
function readMessageText(message: { content: readonly { type: string; text?: string }[] | undefined }): string {
  if (!message?.content) return '';
  return message.content
    .map((part) => (part.type === 'text' ? part.text ?? '' : ''))
    .join('')
    .trim();
}

const AssistantMessage: FC<AssistantMessageProps> = ({ onCitationClick }) => {
  const message = useMessage();
  const customData = (message.metadata?.custom ?? {}) as RagMessageCustomData;
  const sources = customData.sources ?? [];
  const citations = customData.citations ?? [];

  // Resolve the chunk_id → {source, citation} payload at click time
  // so the lookup is fresh even after the message is re-loaded from
  // sessionStorage on reload.
  const handleCitationClick = useCallback(
    (chunkId: string) => {
      const source = sources.find((s) => s.chunk_id === chunkId);
      const citation = citations.find((c) => c.chunk_id === chunkId);
      onCitationClick?.({ chunkId, source, citation });
    },
    [sources, citations, onCitationClick],
  );

  return (
    <CitationClickContext.Provider value={handleCitationClick}>
      <MessagePrimitive.Root className="chat chat-start group">
        <div className="chat-bubble bg-base-100 border border-base-200 text-base-content shadow-sm">
          <MessagePrimitive.Content components={{ Text: MarkdownText }} />
        </div>

        <AssistantActionBar />

        <BranchPicker />
      </MessagePrimitive.Root>
    </CitationClickContext.Provider>
  );
};

/** Action bar for assistant messages — Copy, Reload (regenerate),
 * PDF export, Speak (TTS via WebSpeechSynthesisAdapter wired in
 * runtime.ts). Hidden while the thread is running so it doesn't
 * flicker. */
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
 * branches (created when user edits or regenerates). */
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
