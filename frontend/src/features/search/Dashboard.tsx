/**
 * Dashboard — ECharts charts: Journal Distribution widget, Entity Distribution,
 * Publication Timeline, Plant Origin Heatmap (world map).
 */

import React, { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import { CubeTransparent } from '@phosphor-icons/react';
import { useNavigate } from 'react-router-dom';
import { dashboardApi } from '../../lib/api';
import { ENTITY_COLORS } from '../../lib/entityColors';
import { PHYTOQUERY_THEME_NAME } from '../../lib/echartsTheme';
import { useTheme } from '../../lib/theme';
import JournalDistributionWidget from './JournalDistributionWidget';

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
}: {
  accent: StatAccent;
  label: string;
  value: number;
  footer: React.ReactNode;
}) {
  return (
    <div className="stat-card" data-accent={accent}>
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

// ─── Dashboard ────────────────────────────────────────────────────────────────

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [worldGeo, setWorldGeo] = useState<unknown>(null);
  const navigate = useNavigate();
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [timelineHover, setTimelineHover] = useState(false);

  // Fetch world map geoJSON once (served from /public — no CDN at runtime)
  useEffect(() => {
    fetch('/world.json')
      .then((r) => r.json())
      .then((data) => {
        (echarts as any).registerMap('world', data);
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

  const geoOption: EChartsOption = {
    tooltip: {
      trigger: 'item',
      formatter: (params: any) =>
        params.name
          ? `<b>${params.name}</b><br/><span style="color:#64748b">Papers:</span> ${params.value ?? 0}`
          : '',
    },
    geo: {
      map: 'world',
      roam: true,
      zoom: 1.2,
      itemStyle: {
        areaColor: '#F5F7FA',
        borderColor: '#E2E8F0',
        borderWidth: 0.5,
      },
      emphasis: {
        itemStyle: { areaColor: '#3b82f6' },
        label: { show: false },
      },
    },
    series: [
      {
        type: 'map',
        map: 'world',
        geoIndex: 0,
        data: metrics.charts.geo_distribution.map((d) => ({ name: d.name, value: d.value })),
      },
    ],
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4 py-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 title-font tracking-tight">Database Insights</h2>
          <p className="text-sm text-slate-500 mt-1">Real-time metrics from the gold-standard entity corpus</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate('/dashboard/3d')}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white border border-slate-200 text-sm font-medium text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors shadow-sm"
          >
            <CubeTransparent weight="duotone" size={18} className="text-purple-500" />
            3D View
          </button>
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
        />
      </div>

      {/* Row 1: Journal Distribution Widget (full-width) */}
      <div className="charts-grid mt-8">
        <ChartCard title="Top Journals">
          <JournalDistributionWidget
            journals={metrics.charts.papers_by_journal}
            totalPapers={metrics.kpis.total_papers}
          />
        </ChartCard>

        {/* Row 2: Entity Distribution + Publication Timeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          <ChartCard title="Entity Distribution">
            <div className="h-80 w-full">
              <ReactECharts option={entityDistOption} theme={PHYTOQUERY_THEME_NAME} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
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

        {/* Row 3: Geographic Heatmap */}
        <div className="mt-6">
          <ChartCard title="Plant Origin Heatmap">
            <div className="flex items-center justify-between mb-4">
              <p className="text-xs text-slate-500">Geographic distribution of bioactive species collection sites</p>
              <div className="flex gap-4">
                {metrics.charts.geo_distribution.slice(0, 3).map((item) => (
                  <div key={item.name} className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-600" />
                    <span className="text-xs font-bold text-slate-700">{item.name}: {item.value}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="h-[400px] w-full bg-slate-50/50 rounded-xl overflow-hidden border border-slate-100">
              <ReactECharts option={geoOption} theme={PHYTOQUERY_THEME_NAME} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
            </div>
          </ChartCard>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;