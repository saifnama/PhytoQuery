import React, { useEffect, useState } from 'react';
import { Files, Graph, BookBookmark } from '@phosphor-icons/react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { 
  ComposableMap, 
  Geographies, 
  Geography, 
  ZoomableGroup 
} from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import { dashboardApi } from '../../lib/api';

const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

const colorScale = scaleLinear<string>()
  .domain([0, 10])
  .range(["#eff6ff", "#1e40af"]);

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

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f43f5e', '#6366f1', '#eab308'];

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await dashboardApi.getMetrics();
        setMetrics(data);
      } catch (err) {
        console.error('Failed to fetch dashboard metrics:', err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (isLoading) {
    return (
      <div className="w-full flex justify-center py-12">
        <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
      </div>
    );
  }

  if (!metrics) return null;

  return (
    <div className="w-full max-w-6xl mx-auto px-4 py-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 title-font tracking-tight">Database Insights</h2>
          <p className="text-sm text-slate-500 mt-1">Real-time metrics from the gold-standard entity corpus</p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="saas-card p-6 bg-gradient-to-br from-white to-blue-50/50 border-blue-100">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Total Papers</h3>
            <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center text-blue-600">
              <Files size={24} weight="duotone" />
            </div>
          </div>
          <div className="text-3xl font-bold text-slate-900 mb-1">{metrics.kpis.total_papers.toLocaleString()}</div>
          <div className="text-xs font-medium text-slate-400">Indexed and processed</div>
        </div>

        <div className="saas-card p-6 bg-gradient-to-br from-white to-emerald-50/50 border-emerald-100">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Entities Extracted</h3>
            <div className="w-10 h-10 rounded-xl bg-emerald-100 flex items-center justify-center text-emerald-600">
              <Graph size={24} weight="duotone" />
            </div>
          </div>
          <div className="text-3xl font-bold text-slate-900 mb-1">{metrics.kpis.total_entities.toLocaleString()}</div>
          <div className="text-xs font-medium text-slate-400">Chemicals, species, and locations</div>
        </div>

        <div className="saas-card p-6 bg-gradient-to-br from-white to-purple-50/50 border-purple-100">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Journals Indexed</h3>
            <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center text-purple-600">
              <BookBookmark size={24} weight="duotone" />
            </div>
          </div>
          <div className="text-3xl font-bold text-slate-900 mb-1">{metrics.kpis.total_journals.toLocaleString()}</div>
          <div className="text-xs font-medium text-slate-400 truncate" title={metrics.kpis.top_journals}>
            Top: {metrics.kpis.top_journals}
          </div>
        </div>
      </div>

    <div className="charts-grid mt-8">
      {/* Row 1: Journals and Entity Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Papers by Journal (Bar Chart) */}
        <div className="saas-card p-6">
          <h3 className="text-base font-semibold text-slate-900 mb-6">Top Journals</h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metrics.charts.papers_by_journal} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis 
                  type="category" 
                  dataKey="name" 
                  stroke="#64748b" 
                  fontSize={11} 
                  width={150} 
                  tickLine={false} 
                  axisLine={false} 
                  tickFormatter={(val) => val.length > 20 ? val.substring(0, 20) + '...' : val}
                />
                <RechartsTooltip 
                  cursor={{ fill: '#f8fafc' }} 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Entity Distribution (Pie Chart) */}
        <div className="saas-card p-6">
          <h3 className="text-base font-semibold text-slate-900 mb-6">Entity Distribution</h3>
          <div className="h-64 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={metrics.charts.entity_distribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                  stroke="none"
                >
                  {metrics.charts.entity_distribution.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                  itemStyle={{ color: '#0f172a', fontWeight: 500 }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          {/* Custom Legend */}
          <div className="flex flex-wrap justify-center gap-3 mt-4">
            {metrics.charts.entity_distribution.slice(0, 5).map((entry, index) => (
              <div key={entry.name} className="flex items-center gap-1.5 text-xs text-slate-600 font-medium">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                {entry.name}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row 2: Open Access (Centered or with another chart) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="saas-card p-6 lg:col-span-1">
          <h3 className="text-base font-semibold text-slate-900 mb-6">Open Access</h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={metrics.charts.oa_distribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {metrics.charts.oa_distribution.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.name === 'Open Access' ? '#10b981' : '#f43f5e'} 
                    />
                  ))}
                </Pie>
                <RechartsTooltip 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 mt-4">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
              <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
              Open Access
            </div>
            <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              Restricted
            </div>
          </div>
        </div>
        
        {/* Papers by Year (Histogram) */}
        <div className="saas-card p-6 lg:col-span-2">
          <h3 className="text-base font-semibold text-slate-900 mb-6">Publication Timeline</h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metrics.charts.papers_by_year}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                <RechartsTooltip 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="value" fill="#6366f1" radius={[4, 4, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      {/* Row 3: Geographic Map */}
      <div className="mt-6">
        <div className="saas-card p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-base font-semibold text-slate-900">Plant Origin Heatmap</h3>
              <p className="text-xs text-slate-500 mt-1">Geographic distribution of bioactive species collection sites</p>
            </div>
            <div className="flex gap-4">
              {metrics.charts.geo_distribution.slice(0, 3).map(item => (
                <div key={item.name} className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-600"></span>
                  <span className="text-xs font-bold text-slate-700">{item.name}: {item.value}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="h-[400px] w-full bg-slate-50/50 rounded-xl overflow-hidden border border-slate-100">
            <ComposableMap projectionConfig={{ scale: 140 }}>
              <ZoomableGroup>
                <Geographies geography={geoUrl}>
                  {({ geographies }: { geographies: any[] }) =>
                    geographies.map((geo: any) => {
                      const countryName = geo.properties.name;
                      const d = metrics.charts.geo_distribution.find(
                        (s) => s.name.toLowerCase() === countryName.toLowerCase()
                      );
                      return (
                        <Geography
                          key={geo.rsmKey}
                          geography={geo}
                          fill={d ? colorScale(d.value) : "#F5F7FA"}
                          stroke="#E2E8F0"
                          strokeWidth={0.5}
                          style={{
                            default: { outline: "none" },
                            hover: { fill: "#3b82f6", outline: "none", cursor: "pointer" },
                            pressed: { outline: "none" },
                          }}
                        />
                      );
                    })
                  }
                </Geographies>
              </ZoomableGroup>
            </ComposableMap>
          </div>
        </div>
      </div>
    </div>
    </div>
  );
};

export default Dashboard;
