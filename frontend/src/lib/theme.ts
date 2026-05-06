/**
 * Light/dark theme hook.
 * Strategy:
 *   - Source of truth lives on <html class="dark"> so CSS can react via
 *     `:where(.dark, .dark *)` (Tailwind v4 dark variant + scoped overrides).
 *   - Persisted to localStorage under "phytoquery:theme" as the string
 *     "light" or "dark".
 *   - First load reads localStorage, then falls back to system preference
 *     via prefers-color-scheme.
 *   - Cross-tab sync via the `storage` event so toggling in one tab
 *     updates the others.
 */

import { useCallback, useEffect, useSyncExternalStore } from 'react';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'phytoquery:theme';
const ROOT_DARK_CLASS = 'dark';

function readStored(): Theme | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw === 'light' || raw === 'dark' ? raw : null;
  } catch {
    return null;
  }
}

function readSystem(): Theme {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return 'light';
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function readCurrent(): Theme {
  return readStored() ?? readSystem();
}

function applyToRoot(theme: Theme): void {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  root.classList.toggle(ROOT_DARK_CLASS, theme === 'dark');
  // Hint native form controls + scrollbars to follow our palette.
  root.style.colorScheme = theme;
}

// Apply chosen theme synchronously on first JS evaluation so users don't
// see a flash of the wrong palette while React mounts.
if (typeof document !== 'undefined') {
  applyToRoot(readCurrent());
}

// Tiny subscribe/getSnapshot pair for useSyncExternalStore so every
// component using useTheme stays in sync without a context provider.
const listeners = new Set<() => void>();

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  const onStorage = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY) fn();
  };
  if (typeof window !== 'undefined') {
    window.addEventListener('storage', onStorage);
  }
  return () => {
    listeners.delete(fn);
    if (typeof window !== 'undefined') {
      window.removeEventListener('storage', onStorage);
    }
  };
}

function notify() {
  for (const fn of listeners) fn();
}

function setStored(theme: Theme): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // localStorage may be blocked (private mode); applying still works
    // for the current session.
  }
}

export function useTheme(): {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (next: Theme) => void;
} {
  const theme = useSyncExternalStore(subscribe, readCurrent, () => 'light' as Theme);

  const setTheme = useCallback((next: Theme) => {
    setStored(next);
    applyToRoot(next);
    notify();
  }, []);

  const toggleTheme = useCallback(() => {
    const next: Theme = readCurrent() === 'dark' ? 'light' : 'dark';
    setTheme(next);
  }, [setTheme]);

  // Re-apply on mount in case the DOM was hydrated with a stale className.
  useEffect(() => {
    applyToRoot(theme);
  }, [theme]);

  return { theme, toggleTheme, setTheme };
}
