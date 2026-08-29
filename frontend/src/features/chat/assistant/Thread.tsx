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
  Export,
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
      style={{
        ['--thread-max-width' as string]: '50rem',
        ['--turn-gap-prompt-to-answer' as string]: '36px',
        ['--turn-gap-answer-to-prompt' as string]: '50px',
      }}
    >
      <ThreadPrimitive.Viewport className="relative flex-1 overflow-y-auto px-4 py-8 flex flex-col">
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
      tooltip="Export"
      className="text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-md h-7 w-7 flex items-center justify-center p-0 shadow-none border-0 transition-colors"
    >
      <Export size={14} weight="regular" />
    </TooltipIconButton>
  );
};



const ScrollToBottomButton: FC = () => (
  <ThreadPrimitive.ScrollToBottom asChild>
    <TooltipIconButton
      type="button"
      variant="outline"
      size="icon-sm"
      side="left"
      tooltip="Scroll to latest"
      className="absolute bottom-4 right-4 h-9 w-9 rounded-full bg-background hover:bg-muted text-foreground border border-border/80 shadow-md flex items-center justify-center p-0 disabled:hidden transition-transform active:scale-95"
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
        Upload PDFs in the sidebar, then ask questions.
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
  hr: () => <hr className="my-4 border-0 border-t border-slate-200" />,
  p: ({ children }) => <p className="mb-2.5 last:mb-0 leading-relaxed text-[15px] text-slate-800">{children}</p>,
  h1: ({ children }) => <h1 className="text-base font-bold text-slate-900 mt-3 mb-1.5 first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="text-[15px] font-bold text-slate-900 mt-3 mb-1.5 first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="text-[14px] font-bold text-slate-900 mt-2 mb-1 first:mt-0">{children}</h3>,
  ul: ({ children }) => <ul className="list-disc pl-5 mb-2.5 space-y-1 text-[15px] text-slate-800">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-5 mb-2.5 space-y-1.5 text-[15px] text-slate-800">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
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
      className="w-full text-slate-800 text-[15px] leading-relaxed"
    />
  );
};

const UserMessage: FC = () => {
  const message = useMessage();
  const text = readMessageText(message);

  return (
    <MessagePrimitive.Root className="mx-auto w-full max-w-[var(--thread-max-width)] flex flex-col items-end group animate-in fade-in slide-in-from-bottom-1 duration-150 pt-[var(--turn-gap-answer-to-prompt)] first:pt-0 pb-[var(--turn-gap-prompt-to-answer)]">
      <ComposerPrimitive.If editing>
        <UserEditComposer />
      </ComposerPrimitive.If>

      <ComposerPrimitive.If editing={false}>
        <div className="flex items-center gap-2 w-full justify-end">
          <UserActionBar />

          <div
            className="max-w-[85%] rounded-[22px] px-5 py-2.5 text-[15px] leading-normal text-slate-800 break-words whitespace-pre-wrap select-text"
            style={{ backgroundColor: PINK_USER_BG }}
          >
            {text}
          </div>
        </div>

        <BranchPicker />
      </ComposerPrimitive.If>
    </MessagePrimitive.Root>
  );
};

const UserEditComposer: FC = () => (
  <ComposerPrimitive.Root className="w-full max-w-2xl">
    <div className="flex flex-col gap-2 rounded-2xl border border-border bg-card p-3 shadow-sm">
      <ComposerPrimitive.Input asChild>
        <Textarea
          className="min-h-[60px] resize-none border-0 bg-transparent text-sm shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
          autoFocus
        />
      </ComposerPrimitive.Input>
      <div className="flex justify-end gap-2">
        <ComposerPrimitive.Cancel asChild>
          <Button type="button" variant="ghost" size="sm" className="rounded-full px-3 text-xs font-medium">
            Cancel
          </Button>
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send asChild>
          <Button
            type="submit"
            size="sm"
            className="rounded-full px-4 text-xs font-medium text-white hover:opacity-90 transition-opacity"
            style={{ backgroundColor: PINK_ACCENT }}
          >
            Done
          </Button>
        </ComposerPrimitive.Send>
      </div>
    </div>
  </ComposerPrimitive.Root>
);

const UserActionBar: FC = () => (
  <ActionBarPrimitive.Root
    className="flex items-center opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100 data-[floating=true]:opacity-100"
  >
    <ActionBarPrimitive.Edit asChild>
      <TooltipIconButton
        type="button"
        variant="ghost"
        size="icon-xs"
        tooltip="Edit"
        className="text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-md h-7 w-7 flex items-center justify-center p-0 shadow-none border-0 transition-colors"
      >
        <PencilSimple size={14} weight="regular" />
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
  const text = readMessageText(message);
  const isPending = !text;

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
        {isPending ? (
          <div className="py-2.5 px-1 flex items-center">
            <span className="h-3.5 w-3.5 rounded-full bg-foreground animate-chatgpt-dot" />
          </div>
        ) : (
          <>
            <div className="w-full text-foreground leading-relaxed text-[15px] pt-0">
              <MessagePrimitive.Content components={{ Text: MarkdownText }} />
            </div>

            <AssistantActionBar />

            <BranchPicker />
          </>
        )}
      </MessagePrimitive.Root>
    </CitationClickContext.Provider>
  );
};

const AssistantActionBar: FC = () => (
  <ActionBarPrimitive.Root
    hideWhenRunning
    autohide="not-last"
    autohideFloat="single-branch"
    className="mt-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 data-[floating=true]:opacity-100 data-[autohide=never]:opacity-100"
  >
    <ActionBarPrimitive.Copy asChild>
      <TooltipIconButton
        type="button"
        variant="ghost"
        size="icon-xs"
        tooltip="Copy"
        className="text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-md h-7 w-7 flex items-center justify-center p-0 shadow-none border-0 transition-colors"
      >
        <MessagePrimitive.If copied>
          <Check size={14} weight="bold" className="text-teal-600" />
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
        tooltip="Regenerate"
        className="text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-md h-7 w-7 flex items-center justify-center p-0 shadow-none border-0 transition-colors"
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
  <ComposerPrimitive.Root className="bg-transparent px-4 pb-6 pt-2">
    <div className="mx-auto flex max-w-4xl items-center gap-2.5 rounded-[26px] border border-border/80 bg-background py-1.5 pl-5 pr-2 shadow-xl shadow-black/[0.04] focus-within:border-border transition-all">
      <ComposerPrimitive.Input asChild>
        <Textarea
          rows={1}
          autoFocus
          placeholder="Ask anything..."
          className="min-h-[38px] max-h-[200px] flex-1 resize-none border-0 bg-transparent py-1.5 px-0 text-[15px] leading-6 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/70"
        />
      </ComposerPrimitive.Input>
      <div className="flex items-center shrink-0">
        <ThreadPrimitive.If running>
          <ComposerPrimitive.Cancel asChild>
            <TooltipIconButton
              type="button"
              variant="secondary"
              size="icon"
              side="top"
              tooltip="Stop"
              className="h-9 w-9 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-800 shadow-sm active:scale-95 flex items-center justify-center p-0 shrink-0 border-0"
            >
              <Stop size={14} weight="fill" className="size-3.5" />
            </TooltipIconButton>
          </ComposerPrimitive.Cancel>
        </ThreadPrimitive.If>
        <ThreadPrimitive.If running={false}>
          <ComposerPrimitive.Send asChild>
            <TooltipIconButton
              type="submit"
              size="icon"
              side="top"
              tooltip="Send"
              className="h-9 w-9 rounded-full text-white shadow-sm hover:shadow active:scale-95 disabled:opacity-35 disabled:shadow-none flex items-center justify-center p-0 shrink-0 border-0"
              style={{ backgroundColor: PINK_ACCENT }}
            >
              <ArrowUp size={18} weight="bold" className="size-4.5" />
            </TooltipIconButton>
          </ComposerPrimitive.Send>
        </ThreadPrimitive.If>
      </div>
    </div>
  </ComposerPrimitive.Root>
);
