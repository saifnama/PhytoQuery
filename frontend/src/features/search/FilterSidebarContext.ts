import { createContext } from 'react';

/**
 * Context that signals whether an external filter sidebar is present.
 * When true, SearchForm automatically collapses and hides its internal
 * filter buttons without requiring explicit prop drilling.
 */
export const FilterSidebarContext = createContext<boolean>(false);
