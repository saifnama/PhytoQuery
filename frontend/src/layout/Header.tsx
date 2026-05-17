import React from 'react';
import { Link } from '@tanstack/react-router';
import { Compass, ListMagnifyingGlass, Chats, FlowerTulip } from '@phosphor-icons/react';
import { useTheme } from '../lib/theme';

interface HeaderProps {
  isLoading?: boolean;
}

// Tailwind utility groups for the nav links. `Link.activeProps.className`
// is appended when the link's `to` matches the current pathname, so we get
// type-safe active-state styling without a manual `pathname === '/...'` check.
const NAV_BASE =
  'flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all';
const NAV_INACTIVE = 'text-slate-500 hover:text-slate-700';
const NAV_ACTIVE = 'bg-white text-blue-600 shadow-sm';

const Header: React.FC<HeaderProps> = ({ isLoading = false }) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="h-16 flex items-center px-8 bg-white border-b border-slate-100 sticky top-0 z-30 justify-between">
      <div className="flex items-center space-x-3">
        <button
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          className="theme-toggle grid place-items-center rounded-lg p-1 hover:bg-slate-100 transition-colors"
        >
          <FlowerTulip size={32} color="#db1f83" weight={theme === 'dark' ? 'fill' : 'regular'} />
        </button>
        <Link to="/" className="hover:opacity-80 transition-opacity">
          <span className="text-xl font-bold text-slate-900 title-font">PhytoQuery</span>
        </Link>
      </div>

      <div className="flex items-center space-x-4">
        {isLoading && (
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-slate-200 border-t-blue-600" />
        )}

        <nav className="flex items-center space-x-1 bg-slate-100 p-1 rounded-xl">
          <Link
            to="/"
            // `activeOptions.exact` so /paper/* doesn't keep Search highlighted.
            activeOptions={{ exact: true }}
            className={`${NAV_BASE} ${NAV_INACTIVE}`}
            activeProps={{ className: `${NAV_BASE} ${NAV_ACTIVE}` }}
          >
            <Compass size={18} weight="bold" />
            <span>Explore</span>
          </Link>
          <Link
            to="/analyse"
            className={`${NAV_BASE} ${NAV_INACTIVE}`}
            activeProps={{ className: `${NAV_BASE} ${NAV_ACTIVE}` }}
          >
            <ListMagnifyingGlass size={18} weight="bold" />
            <span>Analyse</span>
          </Link>
          <Link
            to="/chat"
            className={`${NAV_BASE} ${NAV_INACTIVE}`}
            activeProps={{ className: `${NAV_BASE} ${NAV_ACTIVE}` }}
          >
            <Chats size={18} weight="bold" />
            <span>Chat</span>
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
