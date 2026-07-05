import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Link, useRouterState } from '@tanstack/react-router';
import {
  Compass,
  ListMagnifyingGlass,
  Chats,
  Circle,
} from '@phosphor-icons/react';
import { useSearchStore } from '../stores/searchStore';

interface HeaderProps {
  isLoading?: boolean;
}

interface NavItem {
  to: '/' | '/analyse' | '/chat';
  label: string;
  Icon: typeof Compass;
}

const NAV_ITEMS: readonly NavItem[] = [
  { to: '/',        label: 'Explore', Icon: Compass },
  { to: '/analyse', label: 'Analyse', Icon: ListMagnifyingGlass },
  { to: '/chat',    label: 'Chat',    Icon: Chats },
];

const IDLE_COLLAPSE_MS = 10_000;
const SCROLL_COLLAPSE_PX = 80;

/**
 * Header — TulipLogo + PhytoQuery brand on the left, segmented pill nav on
 * the right. The pill collapses to a single solid black circle dot after
 * the user scrolls down past 80px OR after 10s of no scrolling/expand
 * activity. Hovering the dot — or clicking it — pops the pill back open.
 *
 * The sliding indicator behind the active tab is positioned by measuring
 * each Link's bounding box relative to the rail. Resizing observes both
 * the rail and the individual Links so font-loading / window resize
 * never leaves the indicator off the active tab.
 */
const Header: React.FC<HeaderProps> = ({ isLoading = false }) => {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  // Which item is the active tab? — derived from the route.
  const activeIndex =
    pathname.startsWith('/chat')    ? 2 :
    pathname.startsWith('/analyse') ? 1 :
    0;

  // ── collapse state ───────────────────────────────────────────────────
  const [collapsed, setCollapsed] = useState(false);
  const hoveringRef = useRef(false);
  // userOverride = the user manually expanded the dot. Don't re-collapse
  // on small scroll bounces — only on a fresh scroll-down past the
  // threshold OR after the idle timer fires again.
  const userOverrideRef = useRef(false);
  const idleTimerRef = useRef<number | null>(null);

  const resetIdleTimer = useCallback(() => {
    if (idleTimerRef.current !== null) {
      window.clearTimeout(idleTimerRef.current);
    }
    idleTimerRef.current = window.setTimeout(() => {
      if (!hoveringRef.current) {
        userOverrideRef.current = false;
        setCollapsed(true);
      }
    }, IDLE_COLLAPSE_MS);
  }, []);

  // Scroll-driven collapse: down past threshold → collapse;
  // back to top → expand. The idle timer is NOT reset by mouse moves
  // (only by scroll + expand) so the pill always settles to the dot
  // ~10s after it opens.
  useEffect(() => {
    let lastY = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      const goingDown = y > lastY;
      lastY = y;
      resetIdleTimer();
      if (y < 40) {
        userOverrideRef.current = false;
        setCollapsed(false);
        return;
      }
      if (userOverrideRef.current) return;
      if (y > SCROLL_COLLAPSE_PX && goingDown) setCollapsed(true);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    resetIdleTimer();
    return () => {
      window.removeEventListener('scroll', onScroll);
      if (idleTimerRef.current !== null) window.clearTimeout(idleTimerRef.current);
    };
  }, [resetIdleTimer]);

  // ── sliding indicator ────────────────────────────────────────────────
  const railRef = useRef<HTMLDivElement | null>(null);
  const itemRefs = useRef<Array<HTMLAnchorElement | null>>([]);
  const [indicator, setIndicator] = useState({ left: 0, width: 0 });

  const measureIndicator = useCallback(() => {
    const rail = railRef.current;
    const item = itemRefs.current[activeIndex];
    if (!rail || !item) return;
    const railBox = rail.getBoundingClientRect();
    const itemBox = item.getBoundingClientRect();
    setIndicator({
      left:  itemBox.left  - railBox.left,
      width: itemBox.width,
    });
  }, [activeIndex]);

  useEffect(() => {
    if (collapsed) return;
    measureIndicator();
    const ro = new ResizeObserver(measureIndicator);
    if (railRef.current) ro.observe(railRef.current);
    itemRefs.current.forEach((el) => el && ro.observe(el));
    return () => ro.disconnect();
  }, [collapsed, measureIndicator]);

  // ── manual expand (hover/click the dot) ──────────────────────────────
  const expandManual = useCallback(() => {
    hoveringRef.current = true;
    userOverrideRef.current = true;
    setCollapsed(false);
    resetIdleTimer();
  }, [resetIdleTimer]);

  const handlePillLeave = useCallback(() => {
    hoveringRef.current = false;
    if (window.scrollY > SCROLL_COLLAPSE_PX) {
      userOverrideRef.current = false;
      setCollapsed(true);
    } else {
      resetIdleTimer();
    }
  }, [resetIdleTimer]);

  return (
    <header className="app-header">
      <div className="flex items-center gap-[2px]">
        {/* Clicking the brand resets the persisted search store AND
            clears URL search params, so NerPage's `!lastQuery` branch
            kicks in and the Dashboard surfaces. Without the reset,
            sessionStorage-persisted results keep the search view live
            even after navigating to "/". */}
        <Link
          to="/"
          search={{}}
          onClick={() => useSearchStore.getState().resetSearch()}
          className="flex items-center gap-[2px] hover:opacity-80 transition-opacity"
        >
          <img src="/BloomIndex.svg" alt="BloomIndex logo" className="h-[30px] w-auto" />
          <span className="text-[23px] font-bold" style={{ fontFamily: 'var(--font-google-sans, sans-serif)' }}>
            BloomIndex
          </span>
        </Link>
      </div>

      <div className="flex items-center space-x-4">
        {isLoading && (
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-teal-200 border-t-primary" />
        )}

        {collapsed ? (
          <button
            onClick={expandManual}
            onMouseEnter={expandManual}
            title={NAV_ITEMS[activeIndex].label}
            aria-label={`Expand navigation — current: ${NAV_ITEMS[activeIndex].label}`}
            className="grid h-8 w-8 place-items-center bg-transparent text-on-surface border-0 cursor-pointer transition-all duration-200"
            style={{ animation: 'header-pill-fadein .22s ease both' }}
          >
            <Circle size={28} weight="fill" />
          </button>
        ) : (
          <nav
            ref={railRef}
            onMouseEnter={() => {
              hoveringRef.current = true;
              if (idleTimerRef.current !== null) {
                window.clearTimeout(idleTimerRef.current);
              }
            }}
            onMouseLeave={handlePillLeave}
            className="relative inline-flex items-center gap-[2px] p-[4px] bg-[var(--surface-lowest)] border border-[var(--outline-variant)] rounded-full shadow-[0_1px_2px_rgba(0,0,0,0.04)]"
            aria-label="Primary"
          >
            {/* Sliding indicator behind the active tab */}
            <span
              aria-hidden
              className="absolute top-[4px] bottom-[4px] bg-surface-c rounded-full pointer-events-none"
              style={{
                left: indicator.left,
                width: indicator.width,
                transition:
                  'left .32s cubic-bezier(.2,.85,.3,1.1), width .32s cubic-bezier(.2,.85,.3,1.1)',
                zIndex: 0,
              }}
            />
            {NAV_ITEMS.map((item, idx) => {
              const isActive = idx === activeIndex;
              const { Icon } = item;
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  ref={(el) => { itemRefs.current[idx] = el; }}
                  activeOptions={item.to === '/' ? { exact: true } : undefined}
                  style={{ fontFamily: 'var(--font-google-sans)' }}
                  className={[
                    'relative z-[1] inline-flex items-center gap-[8px] h-[40px] px-[18px] rounded-full',
                    'text-[14px] cursor-pointer no-underline transition-colors duration-200',
                    isActive ? 'text-on-surface font-semibold' : 'text-on-surface-variant font-medium hover:text-on-surface',
                  ].join(' ')}
                >
                  <Icon size={18} weight={isActive ? 'bold' : 'regular'} />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>
        )}
      </div>
    </header>
  );
};

export default Header;
