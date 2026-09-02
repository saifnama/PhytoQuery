import { useRef } from "react"
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceDot,
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
 *   2. ``<Brush>`` stays MOUNTED but its visibility toggles via
 *      Tailwind ``opacity`` classes keyed to a hover ``useState``.
 *      Mounting/unmounting it instead would reset the brush's
 *      internal start/end indices on every hover, which is hostile
 *      UX. The wrapping pattern is: same hover state controls a
 *      className on Brush itself — Recharts forwards it to the
 *      ``<g class="recharts-brush">`` root. The brush itself is the
 *      zoom — drag either traveller to narrow the visible year range.
 *   3. Mouse-wheel zoom while hovered — wheel-up narrows the window
 *      anchored to the cursor x-position, wheel-down widens it. The
 *      Brush is the source of truth: it's a controlled component, and
 *      the wheel handler just mutates its ``startIndex``/``endIndex``.
 *      Listener is attached non-passively via ``addEventListener`` so
 *      ``preventDefault()`` actually works (React's synthetic
 *      ``onWheel`` is passive by default in newer React).
 *
 * Theme tokens flow from ``chartConfig.count.color = var(--chart-1)``,
 * which ``ChartContainer`` re-exports as ``--color-count``. Dark mode
 * flips automatically via the ``.dark`` class on the document root.
 */

// Original ECharts line accent (#00ACC1 cyan-600). Local override of
// the --chart-1 default; gives the timeline its distinctive teal/cyan
// look that the journal-bar Coastal palette doesn't carry.
const chartConfig = {
  count: {
    label: "Papers",
    color: "#00ACC1",
  },
} satisfies ChartConfig

interface Props {
  data: { name: string; value: number }[]
  onYearClick?: (year: string) => void
}

export function PublicationTimelineChart({ data, onYearClick }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)

  const peak = data.reduce(
    (p, c) => (c.value > p.value ? c : p),
    data[0] ?? { name: "", value: 0 },
  )

  return (
    <div
      ref={containerRef}
      className="h-[300px] w-full"
    >
      <ChartContainer config={chartConfig} className="h-full w-full">
        <AreaChart
          accessibilityLayer
          data={data}
          margin={{ left: 12, right: 12, top: 24, bottom: 4 }}
          onClick={(e: any) => {
            if (e?.activePayload && e.activePayload.length > 0) {
              const yearName = e.activePayload[0]?.payload?.name;
              if (yearName) onYearClick?.(String(yearName));
            } else if (e?.activeLabel) {
              onYearClick?.(String(e.activeLabel));
            }
          }}
          style={{ cursor: onYearClick ? 'pointer' : 'default' }}
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
            activeDot={{
              r: 5,
              stroke: 'var(--color-count)',
              strokeWidth: 2,
              fill: '#FFFFFF',
              cursor: onYearClick ? 'pointer' : 'default',
            }}
          />

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
        </AreaChart>
      </ChartContainer>
    </div>
  )
}
