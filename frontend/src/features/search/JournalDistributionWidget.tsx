/**
 * Journal Distribution Widget — pure ECharts, Coastal Pastel theme.
 *
 * Two panes:
 *   Left  — dominant journal hero (ECharts `graphic` API, no series).
 *   Right — ranked horizontal bar chart of remaining top journals.
 *
 * Spec: hm.md.
 */

import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import { COASTAL_THEME_NAME, COASTAL_PALETTE } from '../../lib/echartsTheme';

interface JournalEntry {
  name: string;
  value: number;
}

interface Props {
  journals: JournalEntry[];   // sorted desc by paper count, dominant first
  totalPapers: number;        // for percentage calc
  height?: number;            // pane height in px (default 260)
  onJournalClick?: (name: string) => void;  // fires on right-pane bar click
}

const LEFT_PADDING = 28;
const PROGRESS_BAR_WIDTH = 280;

const JournalDistributionWidget: React.FC<Props> = ({ journals, totalPapers, height = 260, onJournalClick }) => {
  if (!journals.length) {
    return (
      <div
        className="flex items-center justify-center text-xs text-slate-400"
        style={{ height }}
      >
        No journal data
      </div>
    );
  }

  const dominant = journals[0];
  const ranked = journals.slice(1, 10);
  const pct = totalPapers > 0 ? Math.round((dominant.value / totalPapers) * 100) : 0;
  const fillWidth = Math.round(
    PROGRESS_BAR_WIDTH * (totalPapers > 0 ? dominant.value / totalPapers : 0)
  );

  // ── Left pane: hero stats via `graphic` API ────────────────────────────────
  const heroOption: EChartsOption = {
    backgroundColor: '#F2FBFC',
    graphic: {
      elements: [
        {
          type: 'text',
          left: LEFT_PADDING,
          top: 28,
          style: {
            text: wrapJournalName(dominant.name),
            font: '500 15px Inter, system-ui, sans-serif',
            fill: '#1A5F6B',
            lineHeight: 22,
          },
        },
        {
          type: 'text',
          left: LEFT_PADDING,
          top: 88,
          style: {
            text: dominant.value.toLocaleString(),
            font: '500 58px Inter, system-ui, sans-serif',
            fill: '#2AACBF',
          },
        },
        {
          type: 'text',
          left: LEFT_PADDING + measureBigNumberWidth(dominant.value) + 8,
          top: 118,
          style: {
            text: 'papers',
            font: '400 15px Inter, system-ui, sans-serif',
            fill: '#5BBCC8',
          },
        },
        {
          type: 'text',
          left: LEFT_PADDING,
          top: 170,
          style: {
            text: `${pct}%`,
            font: '500 28px Inter, system-ui, sans-serif',
            fill: '#2AACBF',
          },
        },
        {
          type: 'rect',
          left: LEFT_PADDING,
          top: 212,
          shape: { width: PROGRESS_BAR_WIDTH, height: 5, r: 3 },
          style: { fill: 'rgba(160,228,241,0.3)' },
        },
        {
          type: 'rect',
          left: LEFT_PADDING,
          top: 212,
          shape: { width: fillWidth, height: 5, r: 3 },
          style: { fill: '#A0E4F1' },
        },
      ],
    },
  };

  // ── Right pane: ranked horizontal bars ─────────────────────────────────────
  const rankedOption: EChartsOption = {
    backgroundColor: '#FBFEFE',
    grid: { top: 10, right: 44, bottom: 10, left: 10, containLabel: true },
    xAxis: {
      type: 'value',
      show: false,
      max: ranked[0]?.value ?? 0,
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: ranked.map((j) => j.name),
      axisLabel: {
        fontSize: 10,
        color: '#5A8080',
        fontFamily: 'Inter, system-ui, sans-serif',
        width: 90,
        overflow: 'truncate',
      },
    },
    series: [
      {
        type: 'bar',
        data: ranked.map((j, i) => ({
          value: j.value,
          itemStyle: {
            color: COASTAL_PALETTE[i % COASTAL_PALETTE.length],
            borderRadius: [0, 3, 3, 0],
          },
        })),
        barMaxWidth: 8,
        label: {
          show: true,
          position: 'right',
          formatter: '{c}',
          fontSize: 10,
          fontFamily: 'Inter, system-ui, sans-serif',
          color: '#7AACAC',
        },
        showBackground: true,
        backgroundStyle: {
          color: 'rgba(160,228,241,0.12)',
          borderRadius: [0, 3, 3, 0],
        },
        emphasis: {
          itemStyle: { opacity: 0.75 },
        },
      },
    ],
    tooltip: {
      show: true,
      backgroundColor: '#fff',
      borderColor: '#A0E4F1',
      borderWidth: 1,
      textStyle: { color: '#1A5F6B', fontSize: 12 },
      formatter: (p: any) => `${p.name}<br/><b>${p.value}</b> papers`,
    },
  };

  return (
    <div
      className="grid overflow-hidden rounded-2xl border"
      style={{
        gridTemplateColumns: '1fr 1px 220px',
        borderColor: '#C8F1F8',
        height,
      }}
    >
      <ReactECharts
        option={heroOption}
        theme={COASTAL_THEME_NAME}
        style={{ width: '100%', height: '100%' }}
        opts={{ renderer: 'canvas' }}
        notMerge
      />
      <div style={{ background: '#C8F1F8' }} />
      <ReactECharts
        option={rankedOption}
        theme={COASTAL_THEME_NAME}
        style={{ width: '100%', height: '100%' }}
        opts={{ renderer: 'canvas' }}
        notMerge
        onEvents={onJournalClick ? {
          click: (params: any) => {
            if (typeof params?.name === 'string') onJournalClick(params.name);
          },
        } : undefined}
      />
    </div>
  );
};

// Wrap journal name to at most two lines of ~24 chars each.
function wrapJournalName(name: string): string {
  if (name.length <= 24) return name;
  const words = name.split(' ');
  const lines: string[] = [];
  let current = '';
  for (const w of words) {
    if ((current + ' ' + w).trim().length > 24 && current) {
      lines.push(current.trim());
      current = w;
      if (lines.length >= 1) break;
    } else {
      current = (current + ' ' + w).trim();
    }
  }
  if (current) lines.push(current.trim());
  return lines.slice(0, 2).join('\n');
}

// Approximate pixel width of the big-number text so we can place "papers"
// beside it without overlap. 58px Inter digits ~ 32px wide; comma ~ 12px.
function measureBigNumberWidth(n: number): number {
  const s = n.toLocaleString();
  let w = 0;
  for (const ch of s) w += ch === ',' ? 12 : 32;
  return w;
}

export default JournalDistributionWidget;
