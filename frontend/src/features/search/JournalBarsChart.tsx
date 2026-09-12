import { useMemo } from "react"
import { Bar, BarChart, Cell, LabelList, XAxis, YAxis } from "recharts"
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

/**
 * Journal Bars — shadcn "Bar Chart - Horizontal" pattern.
 *
 * Implementation notes:
 *   - ``layout="vertical"`` in Recharts means HORIZONTAL bars
 *     (categories on the Y-axis, values on the X-axis). Confusing
 *     naming but it's what the spec demands.
 *   - Bars carry a faint background track via the ``background``
 *     prop on ``<Bar>`` — no extra component, no overlay tricks.
 *   - ``<LabelList position="right" />`` puts the count to the right
 *     of each bar.
 *   - Colour cycles across ``--chart-1`` … ``--chart-5`` via
 *     ``<Cell>``. For >5 entries the last slots reuse early colours,
 *     same approach as EntityDonutChart.
 *   - Y-axis labels truncate at 14 chars to keep the chart compact;
 *     full name still appears in the tooltip.
 */

interface Props {
  data: { name: string; value: number }[]
  onBarClick?: (name: string) => void
}

const DIVERSE_PALETTE = [
  "#0EA5E9", // Sky blue
  "#10B981", // Emerald green
  "#F59E0B", // Amber gold
  "#8B5CF6", // Purple / Violet
  "#EC4899", // Pink
  "#06B6D4", // Cyan
  "#F97316", // Orange
  "#6366F1", // Indigo
  "#14B8A6", // Teal
  "#EF4444", // Coral Red
  "#84CC16", // Lime
  "#D946EF", // Fuchsia
  "#3B82F6", // Cobalt Blue
  "#E11D48", // Rose
  "#22C55E", // Bright Green
  "#A855F7", // Medium Purple
  "#38BDF8", // Light Sky
  "#FB923C", // Light Orange
  "#A78BFA", // Lavender
  "#4ADE80", // Mint
] as const

function truncate(label: string, max = 26): string {
  return label.length > max ? label.slice(0, max - 1) + "…" : label
}

export function JournalBarsChart({ data, onBarClick }: Props) {
  // Stable colour per journal: FNV-1a hash of the name → palette slot.
  // The old code shuffled randomly per data change, so colours jumped on
  // every search/filter. Memoized on data for zero hover cost.
  const dataWithFill = useMemo(() => {
    return data.map((d) => {
      let h = 0x811c9dc5;
      for (let i = 0; i < d.name.length; i++) {
        h ^= d.name.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
      }
      return {
        ...d,
        fill: DIVERSE_PALETTE[(h >>> 0) % DIVERSE_PALETTE.length],
      };
    });
  }, [data]);

  // Single config entry — Recharts only needs the dataKey for tooltip
  // labelling. Per-bar colours are set on ``<Cell>`` below.
  const chartConfig = {
    value: {
      label: "Papers",
      color: "var(--chart-1)",
    },
  } satisfies ChartConfig

  return (
    <ChartContainer
      config={chartConfig}
      className="w-full h-full"
    >
      <BarChart
        accessibilityLayer
        data={dataWithFill}
        layout="vertical"
        margin={{ top: 4, right: 32, bottom: 4, left: 0 }}
      >
        <XAxis type="number" hide />
        <YAxis
          type="category"
          dataKey="name"
          tickLine={false}
          axisLine={false}
          width={195}
          tick={{ fontSize: 11 }}
          tickFormatter={(v: string) => truncate(v, 26)}
        />

        <ChartTooltip
          cursor={false}
          content={
            <ChartTooltipContent
              hideLabel
              formatter={(value, _name, item) => {
                const datum = item.payload as { name: string; value: number }
                return (
                  <div className="flex w-full flex-col gap-1">
                    <span className="text-muted-foreground text-xs">
                      {datum.name}
                    </span>
                    <span className="font-mono font-medium tabular-nums">
                      {Number(value).toLocaleString()} papers
                    </span>
                  </div>
                )
              }}
            />
          }
        />

        <Bar
          dataKey="value"
          radius={[0, 4, 4, 0]}
          background={{
            fill: "var(--muted)",
            opacity: 0.4,
            radius: 4,
          }}
          onClick={(d: unknown) => {
            const payload = (d as { name?: string } | undefined)?.name
            if (payload && onBarClick) onBarClick(payload)
          }}
        >
          {dataWithFill.map((d, i) => (
            <Cell key={`${d.name}-${i}`} fill={d.fill} className="cursor-pointer" />
          ))}
          <LabelList
            dataKey="value"
            position="right"
            className="fill-muted-foreground"
            fontSize={10}
            formatter={(value) => Number(value ?? 0).toLocaleString()}
          />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}
