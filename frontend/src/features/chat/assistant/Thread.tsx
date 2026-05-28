/**
 * assistant-ui chat thread for the PhytoQuery RAG page — shadcn/ui build.
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
  CaretLeft,
  CaretRight,
  X,
  FilePdf,
} from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button';
import { Textarea } from '@/components/ui/textarea';
import {
  Card,
  CardContent,
  CardDescription,
  CardTitle,
} from '@/components/ui/card';
import type { Citation, RagMessageCustomData, RagSource } from './runtime';
import { exportThreadAsPdf, type ThreadTurn } from './exportPdf';

const PINK_ACCENT = '#ff6dba';
const PINK_USER_BG = '#ffecf6';

export interface CitationClickPayload {
  chunkId: string;
  source?: RagSource;
  citation?: Citation;
}

interface ThreadProps {
  onCitationClick?: (payload: CitationClickPayload) => void;
  emptyContent?: ReactNode;
}

const CitationClickContext = createContext<
  ((chunkId: string) => void) | undefined
>(undefined);

export const Thread: FC<ThreadProps> = ({ onCitationClick, emptyContent }) => {
  return (
    <ThreadPrimitive.Root
      className="flex h-full flex-col bg-card"
      style={{ ['--thread-max-width' as string]: '44rem' }}
    >
      <ThreadPrimitive.Viewport className="relative flex-1 overflow-y-auto px-4 py-6 flex flex-col gap-y-6">
        <ThreadPrimitive.Empty>
          <div className="flex h-full items-center justify-center text-muted-foreground">
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

/**
 * Cumulative "export up to here" — clicking the button on the Nth
 * assistant message exports Q+A pairs 1 through N as a single PDF.
 * The button on answer #1 exports just pair 1; on answer #3, pairs 1+2+3.
 * Universal — uses the current message's position in the thread, no
 * hardcoded indices.
 */
const ExportAnswerPdfButton: FC = () => {
  const thread = useThread();
  const message = useMessage();

  const handleExport = () => {
    const messages = thread.messages;
    const idx = messages.findIndex((m) => m.id === message.id);
    if (idx < 0) return;
    // Take every user / assistant turn from the start of the thread
    // through (and including) the assistant message this button sits on.
    const turns: ThreadTurn[] = messages
      .slice(0, idx + 1)
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({
        role: m.role === 'user' ? 'user' : 'assistant',
        text: readMessageText(m),
      }));
    exportThreadAsPdf({ turns });
  };

  return (
    <TooltipIconButton
      type="button"
      variant="ghost"
      size="icon-xs"
      onClick={handleExport}
      tooltip="Export Chat"
    >
      <FilePdf size={14} weight="regular" />
    </TooltipIconButton>
  );
};

/** Typing indicator — three staggered-delay bouncing dots inside a
 * muted bubble. The `animate-typing-dot` keyframe is defined in index.css. */
const TypingIndicator: FC = () => {
  const thread = useThread();
  if (!thread.isRunning) return null;
  return (
    <div className="flex justify-start">
      <div className="inline-flex items-center gap-1 rounded-2xl bg-muted px-4 py-3">
        <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60 animate-typing-dot" />
        <span
          className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60 animate-typing-dot"
          style={{ animationDelay: '0.2s' }}
        />
        <span
          className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60 animate-typing-dot"
          style={{ animationDelay: '0.4s' }}
        />
      </div>
    </div>
  );
};

const ScrollToBottomButton: FC = () => (
  <ThreadPrimitive.ScrollToBottom asChild>
    <TooltipIconButton
      type="button"
      variant="default"
      size="icon-sm"
      side="left"
      tooltip="Scroll to latest"
      className="absolute bottom-4 right-4 rounded-full shadow-md disabled:hidden"
    >
      <ArrowDown size={16} weight="bold" />
    </TooltipIconButton>
  </ThreadPrimitive.ScrollToBottom>
);

const DefaultEmpty: FC = () => (
  <Card size="sm" className="max-w-md ring-0 shadow-none">
    <CardContent className="flex flex-col items-center gap-2 text-center">
      <CardTitle className="text-base">Ask about your papers</CardTitle>
      <CardDescription className="text-center">
        Upload PDFs in the sidebar, then ask questions. Click a citation
        superscript to see the exact passage in the paper.
      </CardDescription>
    </CardContent>
  </Card>
);

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

const markdownComponents = memoizeMarkdownComponents({
  a: CitationLink,
});

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
      const numbering = new Map<string, number>();
      let next = 1;
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
  <MessagePrimitive.Root className="mx-auto w-full max-w-[var(--thread-max-width)] flex flex-col items-end group animate-in fade-in slide-in-from-bottom-1 duration-150">
    <ComposerPrimitive.If editing>
      <UserEditComposer />
    </ComposerPrimitive.If>

    <ComposerPrimitive.If editing={false}>
      <div
        className="inline-block max-w-[80%] rounded-2xl px-4 py-2.5 shadow-sm"
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

const UserEditComposer: FC = () => (
  <ComposerPrimitive.Root className="w-full max-w-2xl">
    <div className="flex flex-col gap-2 rounded-2xl border border-border bg-card p-2 shadow-sm">
      <ComposerPrimitive.Input asChild>
        <Textarea
          className="min-h-[60px] resize-none border-0 bg-transparent text-sm shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
          autoFocus
        />
      </ComposerPrimitive.Input>
      <div className="flex justify-end gap-2">
        <ComposerPrimitive.Cancel asChild>
          <Button type="button" variant="ghost" size="sm">
            <X size={14} weight="bold" />
            Cancel
          </Button>
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send asChild>
          <Button
            type="submit"
            size="sm"
            className="text-white"
            style={{ backgroundColor: PINK_ACCENT }}
          >
            <Check size={14} weight="bold" />
            Update
          </Button>
        </ComposerPrimitive.Send>
      </div>
    </div>
  </ComposerPrimitive.Root>
);

const UserActionBar: FC = () => (
  <ActionBarPrimitive.Root
    autohide="not-last"
    className="mt-1 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 data-[floating=true]:opacity-100"
  >
    <ActionBarPrimitive.Edit asChild>
      <TooltipIconButton
        type="button"
        variant="ghost"
        size="icon-xs"
        tooltip="Edit this question"
      >
        <PencilSimple size={12} weight="regular" />
      </TooltipIconButton>
    </ActionBarPrimitive.Edit>
  </ActionBarPrimitive.Root>
);

interface AssistantMessageProps {
  onCitationClick?: (payload: CitationClickPayload) => void;
}

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
      <MessagePrimitive.Root className="mx-auto w-full max-w-[var(--thread-max-width)] flex flex-col items-start group animate-in fade-in slide-in-from-bottom-1 duration-150">
        <div className="inline-block max-w-[85%] rounded-2xl border border-border bg-card px-4 py-3 text-foreground shadow-sm">
          <MessagePrimitive.Content components={{ Text: MarkdownText }} />
        </div>

        <AssistantActionBar />

        <BranchPicker />
      </MessagePrimitive.Root>
    </CitationClickContext.Provider>
  );
};

const AssistantActionBar: FC = () => (
  <ActionBarPrimitive.Root
    hideWhenRunning
    autohide="not-last"
    autohideFloat="single-branch"
    className="mt-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 data-[floating=true]:opacity-100 data-[autohide=never]:opacity-100"
  >
    <ActionBarPrimitive.Copy asChild>
      <TooltipIconButton
        type="button"
        variant="ghost"
        size="icon-xs"
        tooltip="Copy answer"
      >
        <MessagePrimitive.If copied>
          <Check size={14} weight="bold" className="text-emerald-600" />
        </MessagePrimitive.If>
        <MessagePrimitive.If copied={false}>
          <CopySimple size={14} weight="regular" />
        </MessagePrimitive.If>
      </TooltipIconButton>
    </ActionBarPrimitive.Copy>

    <ActionBarPrimitive.Reload asChild>
      <TooltipIconButton
        type="button"
        variant="ghost"
        size="icon-xs"
        tooltip="Regenerate this answer"
      >
        <ArrowClockwise size={14} weight="regular" />
      </TooltipIconButton>
    </ActionBarPrimitive.Reload>

    <ExportAnswerPdfButton />
  </ActionBarPrimitive.Root>
);

const BranchPicker: FC = () => (
  <MessagePrimitive.If hasBranches>
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className="mt-1 inline-flex items-center gap-0.5 text-xs text-muted-foreground"
    >
      <BranchPickerPrimitive.Previous asChild>
        <TooltipIconButton
          type="button"
          variant="ghost"
          size="icon-xs"
          className="px-1"
          tooltip="Previous branch"
        >
          <CaretLeft size={12} weight="bold" />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="tabular-nums px-1">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton
          type="button"
          variant="ghost"
          size="icon-xs"
          className="px-1"
          tooltip="Next branch"
        >
          <CaretRight size={12} weight="bold" />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  </MessagePrimitive.If>
);

const Composer: FC = () => (
  <ComposerPrimitive.Root className="border-t border-border bg-card p-4">
    <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-[28px] border border-border bg-card px-2 py-1.5 shadow-2xl shadow-black/5 focus-within:border-muted-foreground/40">
      <ComposerPrimitive.Input asChild>
        <Textarea
          rows={1}
          autoFocus
          placeholder="Ask anything..."
          className="min-h-[44px] flex-1 resize-none border-0 bg-transparent text-base shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
        />
      </ComposerPrimitive.Input>
      <div className="pb-1">
        <ThreadPrimitive.If running>
          <ComposerPrimitive.Cancel asChild>
            <TooltipIconButton
              type="button"
              variant="secondary"
              size="icon"
              side="top"
              tooltip="Stop generating"
              className="rounded-full shadow-md active:scale-95"
            >
              <Stop size={18} weight="fill" />
            </TooltipIconButton>
          </ComposerPrimitive.Cancel>
        </ThreadPrimitive.If>
        <ThreadPrimitive.If running={false}>
          <ComposerPrimitive.Send asChild>
            <TooltipIconButton
              type="submit"
              size="icon"
              side="top"
              tooltip="Send message"
              className="rounded-full text-white shadow-md hover:shadow-lg active:scale-95 disabled:opacity-40 disabled:shadow-none"
              style={{ backgroundColor: PINK_ACCENT }}
            >
              <ArrowUp size={18} weight="bold" />
            </TooltipIconButton>
          </ComposerPrimitive.Send>
        </ThreadPrimitive.If>
      </div>
    </div>
  </ComposerPrimitive.Root>
);
