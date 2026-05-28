/**
 * Cross-page signal for opening the Database drawer.
 *
 * The Database drawer (DbExplorerDrawer) is rendered inside Dashboard
 * and owns its own local open/tab/filter state. But other surfaces —
 * notably SearchForm when source=Database is submitted from NerPage —
 * also need to ask the drawer to open with a query pre-applied as the
 * filter.
 *
 * Rather than lift Dashboard's complex drawer state up to NerPage
 * (which would force a structural restructure of Dashboard), we use
 * this tiny ephemeral store as a one-way signal:
 *
 *   - NerPage calls ``requestOpenWithQuery(query)`` when the user
 *     submits the search bar with source=Database.
 *   - Dashboard subscribes to ``pendingOpenQuery``. When it becomes
 *     non-null, Dashboard opens its drawer with that query pushed in
 *     as a search/filter value, then calls ``clearPendingOpenQuery()``
 *     to consume the signal.
 *
 * This state is **not** persisted to sessionStorage — it's a transient
 * intent that should disappear on tab close (and even on remount if
 * not consumed).
 */

import { create } from 'zustand';

interface DrawerState {
  /** Set by NerPage when the user submits the search bar with
   * source=Database. Dashboard reads this on render, opens its
   * drawer with the query pre-applied, then clears the field. */
  pendingOpenQuery: string | null;
  requestOpenWithQuery: (q: string) => void;
  clearPendingOpenQuery: () => void;
}

export const useDrawerStore = create<DrawerState>((set) => ({
  pendingOpenQuery: null,
  requestOpenWithQuery: (q) => set({ pendingOpenQuery: q }),
  clearPendingOpenQuery: () => set({ pendingOpenQuery: null }),
}));
