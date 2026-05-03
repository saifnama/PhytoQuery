/**
 * Dashboard — ECharts charts: Top Journals, Entity Distribution, Open Access,
 * Publication Timeline, Plant Origin Heatmap (world map).
 */

import React, { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import { Files, Graph, BookBookmark, CubeTransparent } from '@phosphor-icons/react';
import { useNavigate } from 'react-router-dom';
import { dashboardApi } from '../../lib/api';
import { ENTITY_COLORS } from '../../lib/entityColors';
import { PHYTOQUERY_THEME_NAME } from '../../lib/echartsTheme';

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
    oa_distribution: { name: string; value: number }[];
    geo_distribution: { name: string; value: number }[];
  };
}

const ENTITY_TYPE_ORDER = [
  'chemical', 'species', 'plant_part', 'extraction_method',
  'analytical_technique', 'bioactivity', 'disease', 'location',
  'season', 'development_stage',
];

const ENTITY_COLOR_LIST = ENTITY_TYPE_ORDER.map(
  (t) => ENTITY_COLORS[t]?.hex ?? ENTITY_COLORS.default.hex
);

// ─── KPI Card ─────────────────────────────────────────────────────────────────

function KpiCard({
  icon,
  label,
  value,
  sub,
  gradient,
  border,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub: string;
  gradient: string;
  border: string;
}) {
  return (
    <div className={`saas-card p-6 bg-gradient-to-br ${gradient} border ${border}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">{label}</h3>
        <div className="w-10 h-10 rounded-xl flex items-center justify-center opacity-80">{icon}</div>
      </div>
      <div className="text-3xl font-bold text-slate-900 mb-1">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
      <div className="text-xs font-medium text-slate-400">{sub}</div>
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

  // ── Chart Options ───────────────────────────────────────────────────────

  const topJournalsOption: EChartsOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 160, right: 40, top: 10, bottom: 10 },
    xAxis: { type: 'value' },
    yAxis: {
      type: 'category',
      data: metrics.charts.papers_by_journal.map((d) => d.name),
      axisLabel: {
        formatter: (v: string) => (v.length > 20 ? v.slice(0, 20) + '…' : v),
      },
    },
    series: [
      {
        type: 'bar',
        data: metrics.charts.papers_by_journal.map((d) => d.value),
        barWidth: 18,
        itemStyle: {
          color: {
            type: 'linear',
            x: 1, y: 0, x2: 0, y2: 0,
            colorStops: [
              { offset: 0, color: '#3b82f6' },
              { offset: 1, color: '#60a5fa' },
            ],
          },
          borderRadius: [0, 4, 4, 0],
        },
        emphasis: { itemStyle: { opacity: 0.8 } },
      },
    ],
  };

  const entityDistOption: EChartsOption = {
    tooltip: { trigger: 'item' },
    legend: { show: false },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '50%'],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
        label: { show: false },
        emphasis: {
          label: { show: true, fontSize: 13, fontWeight: 600, fontFamily: 'Inter' },
          itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.2)' },
        },
        data: metrics.charts.entity_distribution.map((d, i) => ({
          name: d.name,
          value: d.value,
          itemStyle: { color: ENTITY_COLOR_LIST[i % ENTITY_COLOR_LIST.length] },
        })),
      },
    ],
  };

  const oaOption: EChartsOption = {
    tooltip: { trigger: 'item' },
    legend: { show: false },
    series: [
      {
        type: 'pie',
        radius: ['40%', '65%'],
        center: ['50%', '50%'],
        padAngle: 5,
        itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
        label: { show: false },
        data: metrics.charts.oa_distribution.map((d) => ({
          name: d.name,
          value: d.value,
          itemStyle: { color: d.name === 'Open Access' ? '#10b981' : '#f43f5e' },
        })),
      },
    ],
  };

  const timelineOption: EChartsOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 10, bottom: 30 },
    xAxis: {
      type: 'category',
      data: metrics.charts.papers_by_year.map((d) => d.name),
    },
    yAxis: { type: 'value' },
    series: [
      {
        type: 'bar',
        data: metrics.charts.papers_by_year.map((d) => d.value),
        barWidth: 32,
        itemStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: '#6366f1' },
              { offset: 1, color: '#818cf8' },
            ],
          },
          borderRadius: [4, 4, 0, 0],
        },
        emphasis: { itemStyle: { opacity: 0.8 } },
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

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <KpiCard
          icon={<Files size={24} weight="duotone" className="text-blue-600" />}
          label="Total Papers"
          value={metrics.kpis.total_papers}
          sub="Indexed and processed"
          gradient="from-white to-blue-50/50"
          border="border-blue-100"
        />
        <KpiCard
          icon={<Graph size={24} weight="duotone" className="text-emerald-600" />}
          label="Entities Extracted"
          value={metrics.kpis.total_entities}
          sub="Chemicals, species, and locations"
          gradient="from-white to-emerald-50/50"
          border="border-emerald-100"
        />
        <KpiCard
          icon={<BookBookmark size={24} weight="duotone" className="text-purple-600" />}
          label="Journals Indexed"
          value={metrics.kpis.total_journals}
          sub={`Top: ${metrics.kpis.top_journals}`}
          gradient="from-white to-purple-50/50"
          border="border-purple-100"
        />
      </div>

      {/* Row 1: Journals + Entity Distribution */}
      <div className="charts-grid mt-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ChartCard title="Top Journals">
            <div className="h-64 w-full">
              <ReactECharts option={topJournalsOption} theme={PHYTOQUERY_THEME_NAME} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
            </div>
          </ChartCard>

          <ChartCard title="Entity Distribution">
            <div className="h-64 w-full flex items-center justify-center">
              <ReactECharts option={entityDistOption} theme={PHYTOQUERY_THEME_NAME} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
            </div>
            <div className="flex flex-wrap justify-center gap-3 mt-4">
              {metrics.charts.entity_distribution.slice(0, 5).map((entry, i) => (
                <div key={entry.name} className="flex items-center gap-1.5 text-xs text-slate-600 font-medium">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ENTITY_COLOR_LIST[i % ENTITY_COLOR_LIST.length] }} />
                  {entry.name}
                </div>
              ))}
            </div>
          </ChartCard>
        </div>

        {/* Row 2: Open Access + Publication Timeline */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <ChartCard title="Open Access">
            <div className="h-64 w-full">
              <ReactECharts option={oaOption} theme={PHYTOQUERY_THEME_NAME} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
            </div>
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                <div className="w-3 h-3 rounded-full bg-emerald-500" />
                Open Access
              </div>
              <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                Restricted
              </div>
            </div>
          </ChartCard>

          <div className="lg:col-span-2">
            <ChartCard title="Publication Timeline">
              <div className="h-64 w-full">
                <ReactECharts option={timelineOption} theme={PHYTOQUERY_THEME_NAME} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
              </div>
            </ChartCard>
          </div>
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