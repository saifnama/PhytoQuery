/**
 * SunburstChart — ECharts hierarchical sunburst visualization.
 * Used in Dashboard (from /api/dashboard/sunburst) and MyPapersPage (local data).
 */

import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import type { EChartsOption } from 'echarts';
import type { SunburstNode } from '../../types';
import { PHYTOQUERY_THEME_NAME } from '../../lib/echartsTheme';

interface SunburstChartProps {
  data: SunburstNode;
  height?: number;
  onClick?: (node: { name: string; value?: number } | null) => void;
}

function buildLevels(): Array<Record<string, unknown>> {
  return [
    {
      r0: '15%',
      r: '45%',
      itemStyle: { borderWidth: 2, borderColor: '#fff' },
      label: {
        rotate: 'tangential' as const,
        fontSize: 11,
        fontFamily: 'Inter, sans-serif',
        fontWeight: 600,
      },
    },
    {
      r0: '45%',
      r: '75%',
      label: {
        rotate: 'radial' as const,
        fontSize: 12,
        fontFamily: 'Inter, sans-serif',
        fontWeight: 700,
        padding: [0, 0, 5, 0],
      },
      itemStyle: { borderWidth: 2, borderColor: '#fff' },
    },
  ];
}

export const SunburstChart: React.FC<SunburstChartProps> = ({
  data,
  height = 400,
  onClick,
}) => {
  const option = useMemo((): EChartsOption => {
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        formatter: (params: any) => {
          if (!params.treePathInfo) return params.name;
          const path = params.treePathInfo
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            .map((p: any) => p.name)
            .join(' → ');
          return `<b>${params.name}</b><br/>
                  <span style="color:#64748b">Mentions:</span> ${params.value}<br/>
                  <span style="color:#64748b;font-size:10px">${path}</span>`;
        },
      },
      series: {
        type: 'sunburst',
        data: data.children ?? [],
        radius: ['15%', '75%'],
        sort: undefined,
        emphasis: {
          focus: 'ancestor',
        },
        label: {
          show: true,
          fontFamily: 'Inter, sans-serif',
          fontSize: 10,
          color: '#334155',
        },
        itemStyle: {
          borderRadius: 4,
          borderColor: '#fff',
          borderWidth: 1.5,
        },
        levels: buildLevels(),
      },
    };
  }, [data]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const eventsHandlers: any = onClick
    ? { click: (e: unknown) => onClick((e as { data: { name: string; value?: number } }).data || null) }
    : {};

  return (
    <ReactECharts
      option={option}
      theme={PHYTOQUERY_THEME_NAME}
      style={{ height: `${height}px`, width: '100%' }}
      opts={{ renderer: 'canvas' }}
      onEvents={eventsHandlers}
    />
  );
};