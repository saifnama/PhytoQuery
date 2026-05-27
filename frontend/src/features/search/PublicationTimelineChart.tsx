import { useState } from "react"
import {
  Area,
  AreaChart,
  Brush,
  CartesianGrid,
  ReferenceDot,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts"
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

/**
 * Publication Timeline — shadcn "Area Chart - Gradient".
 *
 * Three deliberate departures from the canonical shadcn example:
 *
 *   1. ``<ReferenceDot>`` marks the peak year (was ECharts
 *      ``markPoint: { type: 'max' }``).
 *   2. ``<ReferenceLine>`` shows the corpus-wide average with a
 *      right-aligned label (was ECharts
 *      ``markLine: { type: 'average' }``).
 *   3. ``<Brush>`` stays MOUNTED but its visibility toggles via
 *      Tailwind ``opacity`` classes keyed to a hover ``useState``.
 *      Mounting/unmounting it instead would reset the brush's
 *      internal start/end indices on every hover, which is hostile
 *      UX. The wrapping pattern is: same hover state controls a
 *      className on Brush itself — Recharts forwards it to the
 *      ``<g class="recharts-brush">`` root.
 *
 * Theme tokens flow from ``chartConfig.count.color = var(--chart-1)``,
 * which ``ChartContainer`` re-exports as ``--color-count``. Dark mode
 * flips automatically via the ``.dark`` class on the document root.
 */

const chartConfig = {
  count: {
    label: "Papers",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig

interface Props {
  data: { name: string; value: number }[]
}

export function PublicationTimelineChart({ data }: Props) {
  const [hover, setHover] = useState(false)

  const peak = data.reduce(
    (p, c) => (c.value > p.value ? c : p),
    data[0] ?? { name: "", value: 0 },
  )
  const avg = data.length
    ? Math.round(data.reduce((s, d) => s + d.value, 0) / data.length)
    : 0

  return (
    <div
      className="h-[300px] w-full"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <ChartContainer config={chartConfig} className="h-full w-full">
        <AreaChart
          accessibilityLayer
          data={data}
          margin={{ left: 12, right: 12, top: 12, bottom: 4 }}
        >
          <defs>
            <linearGradient id="fillTimeline" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--color-count)" stopOpacity={0.5} />
              <stop offset="95%" stopColor="var(--color-count)" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <CartesianGrid vertical={false} strokeDasharray="3 3" />

          <XAxis
            dataKey="name"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            interval={4}
          />
          <YAxis tickLine={false} axisLine={false} width={32} />

          <ChartTooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={<ChartTooltipContent indicator="line" />}
          />

          <Area
            dataKey="value"
            name="Papers"
            type="monotone"
            fill="url(#fillTimeline)"
            stroke="var(--color-count)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />

          {/* average reference line */}
          {data.length > 0 && (
            <ReferenceLine
              y={avg}
              stroke="currentColor"
              strokeDasharray="4 4"
              strokeOpacity={0.3}
              label={{
                value: `avg ${avg}`,
                position: "right",
                fill: "currentColor",
                fillOpacity: 0.5,
                fontSize: 10,
                fontFamily: "ui-monospace, monospace",
              }}
            />
          )}

          {/* peak marker */}
          {peak.value > 0 && (
            <ReferenceDot
              x={peak.name}
              y={peak.value}
              r={5}
              fill="var(--color-count)"
              stroke="var(--background)"
              strokeWidth={2}
              ifOverflow="extendDomain"
            />
          )}

          {/* hover-revealed range slider — kept mounted so its internal
              start/end indices survive across hovers */}
          <Brush
            dataKey="name"
            height={18}
            stroke="var(--color-count)"
            travellerWidth={8}
            className={
              "transition-opacity duration-200 " +
              (hover ? "opacity-100" : "opacity-0 pointer-events-none")
            }
          />
        </AreaChart>
      </ChartContainer>
    </div>
  )
}
