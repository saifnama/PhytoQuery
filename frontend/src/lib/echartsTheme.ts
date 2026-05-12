/**
 * PhytoQuery ECharts theme. Captures defaults for tooltip, axes, legend, fonts.
 * Register once at module load; consumers pass theme={PHYTOQUERY_THEME_NAME} to ReactECharts.
 */

import * as echarts from 'echarts';

const FONT = 'Inter, sans-serif';

const axisCommon = {
  axisLine: { show: false },
  axisTick: { show: false },
  axisLabel: { color: '#94a3b8', fontSize: 11, fontFamily: FONT },
  splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' as const } },
};

const theme = {
  textStyle: { fontFamily: FONT, color: '#1f2937' },
  backgroundColor: 'transparent',
  tooltip: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderColor: '#e2e8f0',
    borderWidth: 1,
    borderRadius: 8,
    textStyle: { fontFamily: FONT, fontSize: 12, color: '#1f2937' },
  },
  legend: {
    textStyle: { fontFamily: FONT, fontSize: 11, color: '#475569' },
  },
  categoryAxis: { ...axisCommon, splitLine: { show: false } },
  valueAxis: axisCommon,
  logAxis: axisCommon,
  timeAxis: axisCommon,
};

export const PHYTOQUERY_THEME_NAME = 'phytoquery';

let registered = false;
export function ensurePhytoQueryTheme(): void {
  if (registered) return;
  echarts.registerTheme(PHYTOQUERY_THEME_NAME, theme);
  registered = true;
}

// ─── Coastal Pastel theme (Journal Distribution Widget) ─────────────────────

export const COASTAL_THEME_NAME = 'coastal';

export const COASTAL_PALETTE = [
  '#A0E4F1', '#B9EDB9', '#86CFBF', '#FFCBA5',
  '#A0E4F1', '#B9EDB9', '#86CFBF', '#FFCBA5', '#A0E4F1',
];

const coastalTheme = {
  color: COASTAL_PALETTE,
  backgroundColor: 'transparent',
  textStyle: { fontFamily: FONT },
  categoryAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: { color: '#5A8080', fontSize: 10, fontFamily: FONT },
  },
  valueAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: { show: false },
  },
  bar: {
    itemStyle: { borderRadius: [0, 4, 4, 0] },
  },
};

let coastalRegistered = false;
export function ensureCoastalTheme(): void {
  if (coastalRegistered) return;
  echarts.registerTheme(COASTAL_THEME_NAME, coastalTheme);
  coastalRegistered = true;
}
