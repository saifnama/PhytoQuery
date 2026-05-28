/**
 * Dashboard — fully shadcn-native after PR3:
 *
 *   - Stat cards            → shadcn Card + cva accent variants
 *   - ChartCard wrapper      → shadcn Card + CardHeader + CardTitle
 *   - Journal Distribution   → shadcn Card panes (incl. JournalBarsChart)
 *   - Entity Distribution    → EntityDonutChart (Recharts)
 *   - Publication Timeline   → PublicationTimelineChart (Recharts)
 *   - Plant Origin Heatmap   → PlantOriginMap (react-simple-maps + CSS pulse)
 *
 * No ECharts anywhere — the world map's geo component was the last
 * holdout and is now a CSS-animated SVG via react-simple-maps.
 */

import React, { useEffect, useState } from 'react';
import { dashboardApi } from '../../lib/api';
import JournalDistributionWidget from './JournalDistributionWidget';
import DbExplorerDrawer, { type DrawerTab, type DrawerFilter } from './DbExplorerDrawer';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import { PublicationTimelineChart } from './PublicationTimelineChart';
import { EntityDonutChart } from './EntityDonutChart';
import { PlantOriginMap } from './PlantOriginMap';
import { useDrawerStore } from '../../stores/drawerStore';

// ─── Types ────────────────────────────────────────────────────────────────────

interface DashboardMetrics {
  kpis: {
    total_papers: number;
    total_entities: number;
    total_journals: number;
    top_journals: string;
  };
  charts: {
    papers_by_journal: { name: string; value: number }[];
    entity_distribution: { name: string; value: number }[];
    papers_by_year: { name: string; value: number }[];
    geo_distribution: { name: string; value: number }[];
  };
}

// ─── Stat Card — shadcn Card + cva accent variants ───────────────────────────
//
// Each variant supplies two CSS custom properties (--stat-accent and
// --stat-divider) that the JSX consumes via inline style. This keeps the
// per-accent palette declarative on the component instead of hidden in
// a separate stylesheet, while leaving the shadcn Card's own structural
// classes (bg-card, ring-foreground/10, rounded-2xl, etc.) intact.

const statCardVariants = cva('gap-0 py-5 transition-shadow', {
  variants: {
    accent: {
      aqua: '[--stat-accent:#4DD0E1] [--stat-divider:#B2EBF2]',
      lavender: '[--stat-accent:#9575CD] [--stat-divider:#D1C4E9]',
      sage: '[--stat-accent:#81C784] [--stat-divider:#C8E6C8]',
    },
  },
  defaultVariants: { accent: 'aqua' },
});

interface StatCardProps extends VariantProps<typeof statCardVariants> {
  label: string;
  value: number;
  footer: React.ReactNode;
  onClick?: () => void;
}

function StatCard({ accent, label, value, footer, onClick }: StatCardProps) {
  const clickable = !!onClick;
  return (
    <Card
      className={cn(
        statCardVariants({ accent }),
        clickable && 'cursor-pointer hover:ring-foreground/20',
      )}
      onClick={onClick}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={
        clickable
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
    >
      <CardHeader className="px-5">
        <span
          className="text-[9px] font-medium uppercase tracking-[0.12em]"
          style={{ color: 'var(--stat-accent)' }}
        >
          {label}
        </span>
      </CardHeader>
      <CardContent className="px-5">
        <div className="text-[52px] font-medium font-mono leading-[0.9] tracking-[-0.04em]">
          {value.toLocaleString()}
        </div>
      </CardContent>
      <Separator
        className="mx-5 my-2 [&]:bg-[var(--stat-divider)]"
      />
      <CardFooter className="px-5 text-[11px] text-muted-foreground leading-[1.5]">
        {footer}
      </CardFooter>
    </Card>
  );
}

// ─── Chart Card — shadcn Card composition ────────────────────────────────────
function ChartCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

// Centroid helpers extracted to @/lib/mapCentroids and consumed by
// PlantOriginMap — the dashboard no longer needs them directly.

// ─── Dashboard ────────────────────────────────────────────────────────────────

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // ── DB Explorer drawer state ───────────────────────────────────────────────
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>('papers');
  const [drawerFilter, setDrawerFilter] = useState<DrawerFilter | null>(null);

  const openDrawer = (opts?: { tab?: DrawerTab; filter?: DrawerFilter | null }) => {
    if (opts?.tab) setDrawerTab(opts.tab);
    setDrawerFilter(opts?.filter ?? null);
    setDrawerOpen(true);
  };

  // Cross-page signal: when the user submits the search bar in NerPage
  // with source=Database, that flow calls drawerStore.requestOpenWithQuery(q).
  // We watch that field here and open the drawer with the query pushed
  // in as a paper-tab filter, then clear the signal so it can fire again.
  const pendingOpenQuery = useDrawerStore((s) => s.pendingOpenQuery);
  const clearPendingOpenQuery = useDrawerStore((s) => s.clearPendingOpenQuery);
  useEffect(() => {
    if (!pendingOpenQuery) return;
    openDrawer({
      tab: 'papers',
      filter: {
        kind: 'papers',
        label: `Search: ${pendingOpenQuery}`,
        value: pendingOpenQuery,
      },
    });
    clearPendingOpenQuery();
    // openDrawer is stable in this component; the effect should re-run
    // only when a NEW pendingOpenQuery is published by NerPage.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingOpenQuery]);

  // Fetch metrics. World geo + centroids are now handled inside
  // PlantOriginMap (single source of truth, lazy-loaded with the map
  // component itself).
  useEffect(() => {
    dashboardApi.getMetrics()
      .then(setMetrics)
      .catch((err) => console.error('Failed to fetch dashboard metrics:', err))
      .finally(() => setIsLoading(false));
  }, []);

  const isReady = !!metrics;

  // ── Loading / not-ready guard ──────────────────────────────────────────────
  if (isLoading || !isReady) {
    return (
      <div className="w-full max-w-6xl mx-auto px-4 py-8 space-y-6" aria-label="Loading dashboard">
        {/* Title row */}
        <Skeleton className="h-8 w-48" />
        {/* Stat-card strip — 3 cards */}
        <div className="grid grid-cols-3 gap-3">
          <Skeleton className="h-32 rounded-2xl" />
          <Skeleton className="h-32 rounded-2xl" />
          <Skeleton className="h-32 rounded-2xl" />
        </div>
        {/* Journal Distribution widget */}
        <Skeleton className="h-72 rounded-2xl" />
        {/* Two-up: Entity Distribution + Publication Timeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Skeleton className="h-80 rounded-2xl" />
          <Skeleton className="h-80 rounded-2xl" />
        </div>
        {/* Plant Origin Heatmap */}
        <Skeleton className="h-[520px] rounded-2xl" />
      </div>
    );
  }

  // ── Stat-card derived values ─────────────────────────────────────────────
  const papersTotal = metrics.kpis.total_papers;
  const entitiesTotal = metrics.kpis.total_entities;
  const journalsTotal = metrics.kpis.total_journals;

  const years = metrics.charts.papers_by_year
    .map((d) => Number(d.name))
    .filter((y) => Number.isFinite(y));
  const yearMin = years.length ? Math.min(...years) : null;
  const yearMax = years.length ? Math.max(...years) : null;

  const entitiesPerPaper = papersTotal > 0 ? Math.round(entitiesTotal / papersTotal) : 0;

  const dominant = metrics.charts.papers_by_journal[0];
  const dominantPct = dominant && papersTotal > 0
    ? Math.round((dominant.value / papersTotal) * 100)
    : 0;

  // Entity Distribution + Publication Timeline + Plant Origin Heatmap
  // are all dedicated components now (EntityDonutChart,
  // PublicationTimelineChart, PlantOriginMap). Dashboard.tsx just
  // wires data through them.

  return (
    <div className="w-full max-w-6xl mx-auto px-4 py-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 title-font tracking-tight">Database metrics</h2>
        </div>
      </div>

      {/* KPI Cards — shadcn Card with cva accent variants */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-8">
        <StatCard
          accent="aqua"
          label="Papers indexed"
          value={papersTotal}
          footer={
            yearMin !== null && yearMax !== null ? (
              <>
                {yearMin} <span className="text-[color:var(--stat-accent)] mx-[3px]">→</span> {yearMax}
              </>
            ) : (
              'No year data'
            )
          }
          onClick={() => openDrawer({ tab: 'papers', filter: null })}
        />
        <StatCard
          accent="lavender"
          label="Entities extracted"
          value={entitiesTotal}
          footer={
            <>
              {entitiesPerPaper} entities <span className="text-[color:var(--stat-accent)] mx-[3px]">·</span> per paper
            </>
          }
          onClick={() => openDrawer({ tab: 'entities', filter: null })}
        />
        <StatCard
          accent="sage"
          label="Journals indexed"
          value={journalsTotal}
          footer={
            <>
              1 journal <span className="text-[color:var(--stat-accent)] mx-[3px]">·</span> {dominantPct}% of corpus
            </>
          }
          onClick={() => openDrawer({ tab: 'journals', filter: null })}
        />
      </div>

      {/* Row 1: Journal Distribution Widget (full-width) */}
      <div className="mt-8">
        <ChartCard title="Top Journals">
          <JournalDistributionWidget
            journals={metrics.charts.papers_by_journal}
            totalPapers={metrics.kpis.total_papers}
            onJournalClick={(name) =>
              openDrawer({
                tab: 'journals',
                filter: { kind: 'journal', label: `Journal: ${name}`, value: name },
              })
            }
          />
        </ChartCard>

        {/* Row 2: Entity Distribution + Publication Timeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          <ChartCard title="Entity Distribution">
            <EntityDonutChart
              data={metrics.charts.entity_distribution}
              onSliceClick={(name) =>
                openDrawer({
                  tab: 'entities',
                  filter: { kind: 'entity', label: `Type: ${name}`, value: name },
                })
              }
            />
          </ChartCard>

          <ChartCard title="Publication Timeline">
            <PublicationTimelineChart data={metrics.charts.papers_by_year} />
          </ChartCard>
        </div>

        {/* Row 3: Plant Origin Heatmap — react-simple-maps + CSS pulse */}
        <div className="mt-6">
          <ChartCard title="Geographic distribution of bioactive species collection sites">
            <PlantOriginMap
              data={metrics.charts.geo_distribution}
              onCountryClick={(name) =>
                openDrawer({
                  tab: 'papers',
                  filter: { kind: 'country', label: `Origin: ${name}`, value: name },
                })
              }
            />
          </ChartCard>
        </div>
      </div>

      <DbExplorerDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        tab={drawerTab}
        onTabChange={setDrawerTab}
        filter={drawerFilter}
        onClearFilter={() => setDrawerFilter(null)}
        entities={metrics.charts.entity_distribution}
        journals={metrics.charts.papers_by_journal}
      />
    </div>
  );
};

export default Dashboard;