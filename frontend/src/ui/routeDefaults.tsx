/**
 * Default route UI — used by `createRouter` as fallbacks.
 *
 *  - `DefaultPending`     shows while a route loader is in-flight (route-level
 *                         Suspense). Currently we have no loaders, so this is
 *                         rarely visible — but it'll matter when we add them.
 *  - `DefaultError`       catches errors thrown by any route component.
 *  - `DefaultNotFound`    rendered when the URL matches no route.
 */

import { Link } from '@tanstack/react-router';

export function DefaultPending() {
  return (
    <div className="w-full flex justify-center py-12">
      <div className="w-8 h-8 border-4 border-primary/20 border-t-primary rounded-full animate-spin" />
    </div>
  );
}

export function DefaultError({ error }: { error: Error }) {
  return (
    <div className="w-full max-w-2xl mx-auto px-6 py-16">
      <div className="rounded-2xl border border-red-200 bg-red-50 p-6">
        <h2 className="text-base font-semibold text-red-800 mb-2">
          Something went wrong on this page
        </h2>
        <p className="text-sm text-red-700 mb-4">
          {error?.message ?? 'An unknown error occurred while rendering this route.'}
        </p>
        <Link
          to="/"
          className="inline-flex items-center text-sm font-medium text-red-700 underline hover:text-red-900"
        >
          Back to Search
        </Link>
      </div>
    </div>
  );
}

export function DefaultNotFound() {
  return (
    <div className="w-full max-w-2xl mx-auto px-6 py-24 text-center">
      <p className="text-xs font-mono uppercase tracking-[0.18em] text-on-surface-muted mb-3">
        404
      </p>
      <h1 className="text-2xl font-semibold text-on-surface title-font mb-3">
        Page not found
      </h1>
      <p className="text-sm text-on-surface-variant mb-6">
        The URL you followed does not match any route in this app.
      </p>
      <Link
        to="/"
        className="inline-flex items-center px-4 py-2 rounded-xl bg-primary text-white text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        Back to Search
      </Link>
    </div>
  );
}
