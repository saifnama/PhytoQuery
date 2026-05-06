/**
 * Multi-thread sidebar — daisyUI build.
 *
 * Lists every thread in sessionStorage (via ./threadStore), shows the
 * active one highlighted, and lets the user create / delete / switch.
 * Visual surface uses daisyUI semantic classes (`menu`, `btn`,
 * `bg-base-200`, etc.) so it picks up your daisyUI theme tokens.
 *
 * Title derivation: each row pulls its own first-user-message text out
 * of sessionStorage every render. New threads default to "New chat"
 * until the first message lands.
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
      <aside className="flex w-12 flex-shrink-0 flex-col gap-2 border-r border-base-200 bg-base-200 py-3">
        <button
          type="button"
          onClick={onToggleCollapsed}
          className="btn btn-ghost btn-square btn-sm mx-auto"
          title="Show chats"
          aria-label="Show chats sidebar"
        >
          <SidebarSimple size={18} />
        </button>
        <div className="divider mx-2 my-0" />
        <button
          type="button"
          onClick={onNewThread}
          className="btn btn-ghost btn-square btn-sm mx-auto"
          title="New chat"
          aria-label="New chat"
        >
          <Plus size={18} />
        </button>
      </aside>
    );
  }

  return (
    <aside className="flex w-60 flex-shrink-0 flex-col border-r border-base-200 bg-base-200">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pb-3 pt-5">
        <div className="flex items-center gap-2">
          <ChatCircleDots size={18} weight="duotone" className="text-base-content/60" />
          <h3 className="text-sm font-bold tracking-tight text-base-content">Chats</h3>
        </div>
        <button
          type="button"
          onClick={onToggleCollapsed}
          className="btn btn-ghost btn-square btn-xs"
          title="Hide chats"
          aria-label="Hide chats sidebar"
        >
          <SidebarSimple size={16} />
        </button>
      </div>

      {/* New chat button */}
      <div className="px-3 pb-2">
        <button
          type="button"
          onClick={onNewThread}
          className="btn btn-outline btn-sm w-full gap-2"
        >
          <Plus size={14} weight="bold" />
          <span>New chat</span>
        </button>
      </div>

      {/* Thread rows — daisyUI menu */}
      <ul className="menu menu-sm flex-1 overflow-y-auto px-2 py-1">
        {threads.length === 0 ? (
          <li className="menu-disabled">
            <span className="px-3 py-6 text-center text-xs text-base-content/50">
              No chats yet — start a new one.
            </span>
          </li>
        ) : (
          threads.map((thread) => {
            const isActive = thread.id === activeThreadId;
            const title = deriveTitleFromHistory(thread.id, thread.title);
            return (
              <li key={thread.id} className="group">
                <a
                  onClick={() => onSelectThread(thread.id)}
                  className={`flex items-center justify-between ${
                    isActive ? 'menu-active' : ''
                  }`}
                  title={title}
                >
                  <span className="truncate">{title}</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      onDeleteThread(thread.id);
                    }}
                    className={`btn btn-ghost btn-square btn-xs text-base-content/40 hover:text-error ${
                      isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                    }`}
                    title="Delete chat"
                    aria-label={`Delete chat: ${title}`}
                  >
                    <Trash size={13} />
                  </button>
                </a>
              </li>
            );
          })
        )}
      </ul>
    </aside>
  );
};
