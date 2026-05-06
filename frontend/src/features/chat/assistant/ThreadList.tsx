/**
 * Multi-thread sidebar for the RAG chat page.
 *
 * Lists every thread in sessionStorage (via ./threadStore), shows the
 * active one highlighted, and lets the user create / delete / switch
 * between them. Lives to the LEFT of the existing Sources sidebar so
 * users still have file-checkbox controls visible.
 *
 * Title derivation: each row pulls its own first-user-message text
 * out of sessionStorage every render (cheap; sessionStorage reads are
 * synchronous and small). New threads default to "New chat" until the
 * first message lands.
 */

import { type FC } from 'react';
import { ChatCircleDots, Plus, SidebarSimple, Trash } from '@phosphor-icons/react';
import {
  deriveTitleFromHistory,
  type StoredThread,
} from './threadStore';

interface ThreadListProps {
  threads: readonly StoredThread[];
  activeThreadId: string | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onSelectThread: (id: string) => void;
  onNewThread: () => void;
  onDeleteThread: (id: string) => void;
}

export const ThreadList: FC<ThreadListProps> = ({
  threads,
  activeThreadId,
  collapsed,
  onToggleCollapsed,
  onSelectThread,
  onNewThread,
  onDeleteThread,
}) => {
  if (collapsed) {
    return (
      <aside className="flex w-12 flex-shrink-0 flex-col border-r border-slate-100 bg-slate-50 py-3">
        <button
          type="button"
          onClick={onToggleCollapsed}
          className="mx-auto rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-200 hover:text-slate-700"
          title="Show chats"
        >
          <SidebarSimple size={18} />
        </button>
        <div className="mx-auto mt-2 h-px w-6 bg-slate-200" />
        <button
          type="button"
          onClick={onNewThread}
          className="mx-auto mt-2 rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-200 hover:text-slate-700"
          title="New chat"
        >
          <Plus size={18} />
        </button>
      </aside>
    );
  }

  return (
    <aside className="flex w-60 flex-shrink-0 flex-col border-r border-slate-100 bg-slate-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pb-3 pt-5">
        <div className="flex items-center gap-2">
          <ChatCircleDots size={18} weight="duotone" className="text-slate-400" />
          <h3 className="text-sm font-bold tracking-tight text-slate-800">Chats</h3>
        </div>
        <button
          type="button"
          onClick={onToggleCollapsed}
          className="rounded p-1 text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-600"
          title="Hide chats"
        >
          <SidebarSimple size={16} />
        </button>
      </div>

      {/* New chat button */}
      <div className="px-3 pb-2">
        <button
          type="button"
          onClick={onNewThread}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm ring-1 ring-slate-200 transition-all hover:bg-slate-100"
        >
          <Plus size={14} weight="bold" />
          <span>New chat</span>
        </button>
      </div>

      {/* Thread rows */}
      <div className="flex-1 overflow-y-auto px-2 py-1">
        {threads.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-slate-400">
            No chats yet — start a new one.
          </div>
        ) : (
          threads.map((thread) => {
            const isActive = thread.id === activeThreadId;
            const title = deriveTitleFromHistory(thread.id, thread.title);
            return (
              <div
                key={thread.id}
                onClick={() => onSelectThread(thread.id)}
                className={`group flex cursor-pointer items-center justify-between rounded-md px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? 'bg-white font-medium text-slate-900 shadow-sm ring-1 ring-slate-200'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                }`}
                title={title}
              >
                <span className="truncate">{title}</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteThread(thread.id);
                  }}
                  className={`ml-2 rounded p-1 text-slate-300 transition-all hover:bg-red-50 hover:text-red-500 ${
                    isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                  }`}
                  title="Delete chat"
                  aria-label={`Delete chat: ${title}`}
                >
                  <Trash size={13} />
                </button>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
};
