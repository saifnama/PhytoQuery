import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { MagnifyingGlass, Chats, FlowerTulip } from '@phosphor-icons/react';

interface HeaderProps {
  isLoading?: boolean;
}

const Header: React.FC<HeaderProps> = ({
  isLoading = false,
}) => {
  const location = useLocation();
  const navigate = useNavigate();
  const isSearch = location.pathname === '/';
  const isChat = location.pathname === '/chat';

  return (
    <header className="h-16 flex items-center px-8 bg-white border-b border-slate-100 sticky top-0 z-30 justify-between">
      <div className="flex items-center space-x-3">
        <FlowerTulip size={32} color="#db1f83" />
        <span className="text-xl font-bold text-slate-900 title-font">PhytoQuery</span>
      </div>

      <div className="flex items-center space-x-4">
        {isLoading && (
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-slate-200 border-t-blue-600" />
        )}

        <nav className="flex items-center space-x-1 bg-slate-100 p-1 rounded-xl">
          <button
            onClick={() => navigate('/')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              isSearch
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            <MagnifyingGlass size={18} weight="bold" />
            <span>Search</span>
          </button>
          <button
            onClick={() => navigate('/chat')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              isChat
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            <Chats size={18} weight="bold" />
            <span>Chat</span>
          </button>
        </nav>
      </div>
    </header>
  );
};

export default Header;
