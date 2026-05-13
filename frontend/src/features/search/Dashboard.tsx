/**
 * Dashboard — ECharts charts: Journal Distribution widget, Entity Distribution,
 * Publication Timeline, Plant Origin Heatmap (world map).
 */

import React, { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import { dashboardApi } from '../../lib/api';
import { ENTITY_COLORS } from '../../lib/entityColors';
import { PHYTOQUERY_THEME_NAME } from '../../lib/echartsTheme';
import { useTheme } from '../../lib/theme';
import JournalDistributionWidget from './JournalDistributionWidget';
import DbExplorerDrawer, { type DrawerTab, type DrawerFilter } from './DbExplorerDrawer';

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

// ─── Entity distribution helpers (entity_donut_v2_prompt.md spec) ───────────

function entityKey(displayName: string): string {
  return displayName.toLowerCase().replace(/\s+/g, '_');
}

function formatEntityName(displayName: string): string {
  return displayName.replace(/_/g, ' ');
}

interface EntityDatum {
  value: number;
  name: string;
  itemStyle: { color: string };
}

function prepareEntityData(
  raw: { name: string; value: number }[]
): EntityDatum[] {
  return raw.map((d) => {
    const key = entityKey(d.name);
    return {
      value: d.value,
      name: formatEntityName(d.name),
      itemStyle: { color: ENTITY_COLORS[key]?.hex ?? '#94A3B8' },
    };
  });
}

// ─── Stat Card (Style 5 — spec: h.md) ─────────────────────────────────────────

type StatAccent = 'aqua' | 'lavender' | 'sage';

function StatCard({
  accent,
  label,
  value,
  footer,
  onClick,
}: {
  accent: StatAccent;
  label: string;
  value: number;
  footer: React.ReactNode;
  onClick?: () => void;
}) {
  const clickable = !!onClick;
  return (
    <div
      className="stat-card"
      data-accent={accent}
      onClick={onClick}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={
        clickable
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClick!();
              }
            }
          : undefined
      }
      style={clickable ? { cursor: 'pointer' } : undefined}
    >
      <div className="stat-label">{label}</div>
      <div className="stat-number">{value.toLocaleString()}</div>
      <div className="stat-divider" />
      <div className="stat-footer">{footer}</div>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="saas-card p-6">
      <h3 className="text-base font-semibold text-slate-900 mb-6">{title}</h3>
      {children}
    </div>
  );
}

// ─── Country centroid helpers (geo_light_prompt.md spec) ────────────────────

type Lnglat = [number, number];

// Aliases map NER variants to the EXACT canonical names used in /public/world.json
// (which uses short English names: "Russia", "Turkey", "Iran", "Vietnam", "Korea",
// "Czech Rep.", "Lao PDR", "Dem. Rep. Korea", "Dem. Rep. Congo").
const COUNTRY_ALIASES: Record<string, string> = {
  'usa': 'United States',
  'u.s.a.': 'United States',
  'u.s.': 'United States',
  'us': 'United States',
  'united states of america': 'United States',
  'america': 'United States',
  'uk': 'United Kingdom',
  'u.k.': 'United Kingdom',
  'britain': 'United Kingdom',
  'great britain': 'United Kingdom',
  'russian federation': 'Russia',
  'türkiye': 'Turkey',
  'turkiye': 'Turkey',
  'iran (islamic republic of)': 'Iran',
  'syrian arab republic': 'Syria',
  'viet nam': 'Vietnam',
  'venezuela (bolivarian republic of)': 'Venezuela',
  'bolivia (plurinational state of)': 'Bolivia',
  'united republic of tanzania': 'Tanzania',
  'republic of moldova': 'Moldova',
  'north macedonia': 'Macedonia',
  'south korea': 'Korea',
  'republic of korea': 'Korea',
  'north korea': 'Dem. Rep. Korea',
  "korea, democratic people's republic of": 'Dem. Rep. Korea',
  'czech republic': 'Czech Rep.',
  'czechia': 'Czech Rep.',
  'laos': 'Lao PDR',
  "lao people's democratic republic": 'Lao PDR',
  'drc': 'Dem. Rep. Congo',
  'democratic republic of congo': 'Dem. Rep. Congo',
  'democratic republic of the congo': 'Dem. Rep. Congo',
  'ivory coast': "Côte d'Ivoire",
  "cote d'ivoire": "Côte d'Ivoire",
  'cote d ivoire': "Côte d'Ivoire",
};

// GeoJSON in the wild is messy — empty rings, NaN coords, GeometryCollection,
// missing properties. Every helper below treats malformed input as "skip,
// don't throw" so one bad feature can never poison the centroid map.

function isLnglatPoint(p: unknown): p is [number, number] {
  return (
    Array.isArray(p) &&
    p.length >= 2 &&
    typeof p[0] === 'number' && Number.isFinite(p[0]) &&
    typeof p[1] === 'number' && Number.isFinite(p[1])
  );
}

function isRing(r: unknown): r is [number, number][] {
  return Array.isArray(r) && r.length > 0 && r.every(isLnglatPoint);
}

function ringArea(ring: [number, number][]): number {
  if (ring.length < 3) return 0;
  let a = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
  }
  return Math.abs(a / 2);
}

function ringBboxCenter(ring: [number, number][]): Lnglat | null {
  let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
  for (const [x, y] of ring) {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  if (!Number.isFinite(xMin) || !Number.isFinite(yMin)) return null;
  const cx = (xMin + xMax) / 2;
  const cy = (yMin + yMax) / 2;
  return Number.isFinite(cx) && Number.isFinite(cy) ? [cx, cy] : null;
}

// Walks any GeoJSON geometry — Polygon, MultiPolygon, or GeometryCollection —
// and returns every valid outer ring it can find.
function collectOuterRings(geom: any): [number, number][][] {
  const rings: [number, number][][] = [];
  if (!geom || typeof geom !== 'object') return rings;
  const t = geom.type;
  if (t === 'Polygon') {
    const ring = geom.coordinates?.[0];
    if (isRing(ring)) rings.push(ring);
  } else if (t === 'MultiPolygon') {
    for (const poly of geom.coordinates ?? []) {
      const ring = poly?.[0];
      if (isRing(ring)) rings.push(ring);
    }
  } else if (t === 'GeometryCollection') {
    for (const sub of geom.geometries ?? []) {
      rings.push(...collectOuterRings(sub));
    }
  }
  return rings;
}

function featureCentroid(feature: any): Lnglat | null {
  const rings = collectOuterRings(feature?.geometry);
  if (rings.length === 0) return null;
  // Largest polygon (by area) wins — keeps centroids on the main landmass
  // for multi-island countries instead of drifting offshore.
  let largest = rings[0];
  let maxArea = ringArea(largest);
  for (let i = 1; i < rings.length; i++) {
    const a = ringArea(rings[i]);
    if (a > maxArea) { maxArea = a; largest = rings[i]; }
  }
  return ringBboxCenter(largest);
}

function buildCentroidMap(world: any): Map<string, Lnglat> {
  const map = new Map<string, Lnglat>();
  const features = Array.isArray(world?.features) ? world.features : [];
  let failed = 0;
  for (const f of features) {
    try {
      const name = f?.properties?.name;
      if (typeof name !== 'string' || !name.trim()) continue;
      const c = featureCentroid(f);
      if (c) map.set(name.toLowerCase(), c);
    } catch (err) {
      failed += 1;
      // One bad feature must not break the rest of the map.
    }
  }
  if (failed > 0) {
    console.warn(`[Dashboard] centroid map skipped ${failed} malformed feature(s)`);
  }
  return map;
}

function lookupCentroid(rawName: string, centroids: Map<string, Lnglat>): Lnglat | null {
  const key = rawName.trim().toLowerCase();
  const direct = centroids.get(key);
  if (direct) return direct;
  const aliased = COUNTRY_ALIASES[key];
  return aliased ? centroids.get(aliased.toLowerCase()) ?? null : null;
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [worldGeo, setWorldGeo] = useState<unknown>(null);
  const [centroids, setCentroids] = useState<Map<string, Lnglat>>(() => new Map());
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [timelineHover, setTimelineHover] = useState(false);

  // ── DB Explorer drawer state ───────────────────────────────────────────────
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>('papers');
  const [drawerFilter, setDrawerFilter] = useState<DrawerFilter | null>(null);

  const openDrawer = (opts?: { tab?: DrawerTab; filter?: DrawerFilter | null }) => {
    if (opts?.tab) setDrawerTab(opts.tab);
    setDrawerFilter(opts?.filter ?? null);
    setDrawerOpen(true);
  };

  // Fetch world map geoJSON once (served from /public — no CDN at runtime)
  useEffect(() => {
    fetch('/world.json')
      .then((r) => r.json())
      .then((data) => {
        (echarts as any).registerMap('world', data);
        setCentroids(buildCentroidMap(data));
        setWorldGeo(data);
      })
      .catch(() => {
        setWorldGeo({});
      });
  }, []);

  // Fetch metrics
  useEffect(() => {
    dashboardApi.getMetrics()
      .then(setMetrics)
      .catch((err) => console.error('Failed to fetch dashboard metrics:', err))
      .finally(() => setIsLoading(false));
  }, []);

  const isReady = !!metrics && !!worldGeo;

  // ── Loading / not-ready guard ──────────────────────────────────────────────
  if (isLoading || !isReady) {
    return (
      <div className="w-full flex justify-center py-12">
        <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
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

  // ── Chart Options ───────────────────────────────────────────────────────

  const entityDistOption: EChartsOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {d}%',
    },
    legend: {
      top: '5%',
      left: 'center',
      icon: 'circle',
      itemWidth: 8,
      itemHeight: 8,
      itemGap: 12,
      textStyle: { color: '#90A4AE', fontSize: 11 },
      selectedMode: false,
    },
    series: [
      {
        name: 'Entity Distribution',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        label: { show: false, position: 'center' },
        emphasis: {
          label: {
            show: true,
            fontSize: 28,
            fontWeight: 'bold',
            color: '#1A5F6B',
            formatter: '{b}',
          },
          itemStyle: {
            shadowBlur: 20,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.25)',
          },
        },
        labelLine: { show: false },
        itemStyle: {
          borderRadius: 6,
          borderColor: 'transparent',
          borderWidth: 2,
        },
        data: prepareEntityData(metrics.charts.entity_distribution),
      },
    ],
  };

  // ── Publication Timeline — smooth area chart (spec: publication_timeline_prompt.md) ──
  const timelineYears = metrics.charts.papers_by_year.map((d) => d.name);
  const timelineCounts = metrics.charts.papers_by_year.map((d) => d.value);
  const timelineAvg = timelineCounts.length
    ? Math.round(timelineCounts.reduce((a, b) => a + b, 0) / timelineCounts.length)
    : 0;

  const timelineOption: EChartsOption = {
    backgroundColor: 'transparent',
    grid: { top: 28, right: 16, bottom: 60, left: 48, containLabel: false },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'line',
        lineStyle: { color: '#4DD0E1', width: 1, type: 'dashed' },
      },
      backgroundColor: isDark ? '#1E2535' : '#ffffff',
      borderColor: isDark ? 'rgba(255,255,255,0.1)' : '#E0E0E0',
      borderWidth: 1,
      textStyle: {
        color: isDark ? '#F0F2F8' : '#111111',
        fontSize: 12,
        fontFamily: 'Inter, system-ui, sans-serif',
      },
      formatter: (params: any) => {
        const p = Array.isArray(params) ? params[0] : params;
        return `<b style="font-family:monospace">${p.axisValue}</b><br/>${p.value} papers`;
      },
    },
    xAxis: {
      type: 'category',
      data: timelineYears,
      boundaryGap: false,
      axisLine: { lineStyle: { color: isDark ? 'rgba(255,255,255,0.1)' : '#E0E0E0' } },
      axisTick: { show: false },
      axisLabel: {
        color: isDark ? '#7880A0' : '#6B7280',
        fontSize: 10,
        fontFamily: 'monospace',
        interval: 4,
        margin: 10,
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: isDark ? '#7880A0' : '#6B7280',
        fontSize: 10,
        fontFamily: 'monospace',
      },
      splitLine: {
        lineStyle: {
          color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
        },
      },
    },
    dataZoom: [
      {
        type: 'slider',
        show: timelineHover,
        height: 18,
        bottom: 8,
        borderColor: 'transparent',
        backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#F0F0F0',
        fillerColor: isDark ? 'rgba(77,208,225,0.2)' : 'rgba(77,208,225,0.15)',
        handleStyle: { color: '#4DD0E1' },
        moveHandleStyle: { color: '#4DD0E1' },
        textStyle: { color: isDark ? '#7880A0' : '#6B7280', fontSize: 10 },
        labelFormatter: (val: number) => timelineYears[val] ?? '',
      },
      { type: 'inside' },
    ],
    series: [
      {
        type: 'line',
        data: timelineCounts,
        smooth: 0.4,
        symbol: 'none',
        symbolSize: 6,
        lineStyle: { color: '#00ACC1', width: 2 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: isDark ? 'rgba(0,172,193,0.4)' : 'rgba(77,208,225,0.35)' },
              { offset: 1, color: isDark ? 'rgba(0,172,193,0.02)' : 'rgba(77,208,225,0.02)' },
            ],
          },
        },
        markPoint: {
          symbol: 'circle',
          symbolSize: 8,
          itemStyle: {
            color: '#00ACC1',
            borderColor: isDark ? '#1E2535' : '#ffffff',
            borderWidth: 2,
          },
          label: { show: false },
          data: [{ type: 'max', name: 'Peak' }],
        },
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: {
            color: isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)',
            type: 'dashed',
            width: 1,
          },
          label: {
            show: true,
            position: 'end',
            formatter: `avg ${timelineAvg}`,
            color: isDark ? '#7880A0' : '#6B7280',
            fontSize: 10,
            fontFamily: 'monospace',
          },
          data: [{ type: 'average', name: 'Average' }],
        },
      },
    ],
  };

  // ── Plant Origin Heatmap (geo_light_prompt.md — Ghost Map effectScatter) ──
  const geoPoints = metrics.charts.geo_distribution
    .map((d) => {
      const c = lookupCentroid(d.name, centroids);
      if (!c) return null;
      return { name: d.name, value: [c[0], c[1], d.value] as [number, number, number] };
    })
    .filter((p): p is { name: string; value: [number, number, number] } => p !== null);

  const geoMax = geoPoints.reduce((m, p) => Math.max(m, p.value[2]), 1);

  const geoOption: EChartsOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(255,255,255,0.97)',
      borderColor: '#E2E8F0',
      borderWidth: 1,
      extraCssText:
        'border-radius:8px;padding:8px 12px;font-family:Inter,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,0.08)',
      formatter: (p: any) => {
        const v = Array.isArray(p.value) ? p.value[2] : p.value;
        return `<b style="color:#0E7490;font-size:11px">${p.name}</b><br/><span style="color:#8A95B0;font-size:10px">${v} papers</span>`;
      },
    },
    geo: {
      map: 'world',
      roam: true,
      itemStyle: {
        areaColor: '#FFFFFF',
        borderColor: '#D8E8EE',
        borderWidth: 0.6,
      },
      emphasis: {
        itemStyle: {
          areaColor: '#EBF8FB',
          borderColor: '#9DE4EF',
          borderWidth: 0.8,
        },
        label: { show: false },
      },
    },
    series: [
      {
        type: 'effectScatter',
        coordinateSystem: 'geo',
        data: geoPoints,
        encode: { value: 2 },
        symbolSize: (d: number[]) => Math.max(5, Math.sqrt(d[2] / geoMax) * 33),
        rippleEffect: { brushType: 'stroke', scale: 3.3, period: 3.3 },
        itemStyle: { color: '#06B6D4', opacity: 0.92 },
        emphasis: {
          itemStyle: {
            opacity: 1,
            shadowBlur: 14,
            shadowColor: 'rgba(6,182,212,0.35)',
          },
        },
        label: { show: false },
      },
    ],
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4 py-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 title-font tracking-tight">Database metrics</h2>
        </div>
      </div>

      {/* KPI Cards — Style 5 (h.md) */}
      <div className="stats-strip mb-8">
        <StatCard
          accent="aqua"
          label="Papers indexed"
          value={papersTotal}
          footer={
            yearMin !== null && yearMax !== null ? (
              <>
                {yearMin} <span className="stat-accent">→</span> {yearMax}
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
              {entitiesPerPaper} entities <span className="stat-accent">·</span> per paper
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
              1 journal <span className="stat-accent">·</span> {dominantPct}% of corpus
            </>
          }
          onClick={() => openDrawer({ tab: 'journals', filter: null })}
        />
      </div>

      {/* Row 1: Journal Distribution Widget (full-width) */}
      <div className="charts-grid mt-8">
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
            <div className="h-80 w-full">
              <ReactECharts
                option={entityDistOption}
                theme={PHYTOQUERY_THEME_NAME}
                style={{ width: '100%', height: '100%' }}
                opts={{ renderer: 'canvas' }}
                onEvents={{
                  click: (params: any) => {
                    if (typeof params?.name === 'string') {
                      openDrawer({
                        tab: 'entities',
                        filter: { kind: 'entity', label: `Type: ${params.name}`, value: params.name },
                      });
                    }
                  },
                }}
              />
            </div>
          </ChartCard>

          <ChartCard title="Publication Timeline">
            <div
              className="h-[300px] w-full"
              onMouseEnter={() => setTimelineHover(true)}
              onMouseLeave={() => setTimelineHover(false)}
            >
              <ReactECharts option={timelineOption} theme={PHYTOQUERY_THEME_NAME} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
            </div>
          </ChartCard>
        </div>

        {/* Row 3: Geographic Heatmap — Ghost Map (geo_light_prompt.md) */}
        <div className="mt-6">
          <ChartCard title="Geographic distribution of bioactive species collection sites">
            <div
              className="h-[520px] w-full rounded-2xl overflow-hidden"
              style={{ background: '#EDF5F8' }}
            >
              <ReactECharts
                option={geoOption}
                theme={PHYTOQUERY_THEME_NAME}
                style={{ width: '100%', height: '100%' }}
                opts={{ renderer: 'canvas' }}
                onEvents={{
                  click: (params: any) => {
                    // effectScatter dots emit seriesType: 'effectScatter' with a name (country).
                    // Geo-region clicks (the country fill itself) also pass params.name.
                    if (typeof params?.name === 'string' && params.name) {
                      openDrawer({
                        tab: 'papers',
                        filter: { kind: 'country', label: `Origin: ${params.name}`, value: params.name },
                      });
                    }
                  },
                }}
              />
            </div>
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