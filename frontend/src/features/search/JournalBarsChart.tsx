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

const PALETTE = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
] as const

const ROW_HEIGHT = 28

function truncate(label: string, max = 14): string {
  return label.length > max ? label.slice(0, max - 1) + "…" : label
}

export function JournalBarsChart({ data, onBarClick }: Props) {
  const dataWithFill = useMemo(
    () => data.map((d, i) => ({ ...d, fill: PALETTE[i % PALETTE.length] })),
    [data],
  )

  // Single config entry — Recharts only needs the dataKey for tooltip
  // labelling. Per-bar colours are set on ``<Cell>`` below.
  const chartConfig = {
    value: {
      label: "Papers",
      color: "var(--chart-1)",
    },
  } satisfies ChartConfig

  const height = Math.max(180, data.length * ROW_HEIGHT + 32)

  return (
    <ChartContainer
      config={chartConfig}
      style={{ height }}
      className="w-full"
    >
      <BarChart
        accessibilityLayer
        data={dataWithFill}
        layout="vertical"
        margin={{ top: 8, right: 44, bottom: 8, left: 4 }}
      >
        <XAxis type="number" hide />
        <YAxis
          type="category"
          dataKey="name"
          tickLine={false}
          axisLine={false}
          width={120}
          tick={{ fontSize: 11 }}
          tickFormatter={(v: string) => truncate(v)}
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
