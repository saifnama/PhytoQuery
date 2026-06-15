import { useEffect, useRef, useState } from "react"
import {
  Area,
  AreaChart,
  Brush,
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
}

export function PublicationTimelineChart({ data }: Props) {
  const [hover, setHover] = useState(false)
  const containerRef = useRef<HTMLDivElement | null>(null)

  // Brush window — controlled. Wheel handler mutates this; the Brush
  // also writes back via onChange when the user drags its travellers.
  const lastIdx = Math.max(0, data.length - 1)
  const [range, setRange] = useState({ startIndex: 0, endIndex: lastIdx })

  // Reset the window whenever the dataset changes shape (e.g. year
  // count grows from a re-ingest). Without this, an old endIndex could
  // point past the new array.
  useEffect(() => {
    setRange({ startIndex: 0, endIndex: Math.max(0, data.length - 1) })
  }, [data.length])

  // Wheel-driven zoom. React's onWheel is attached passively in modern
  // React, so preventDefault() is a no-op there — we need to bind the
  // listener ourselves with { passive: false } to stop the page from
  // scrolling while the user is zooming the chart.
  useEffect(() => {
    const el = containerRef.current
    if (!el || data.length < 2) return

    const handler = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))

      setRange((prev) => {
        const span = prev.endIndex - prev.startIndex
        // wheel-up (deltaY < 0) zooms in; wheel-down zooms out.
        const factor = e.deltaY < 0 ? 0.85 : 1.18
        const maxSpan = data.length - 1
        const newSpan = Math.max(1, Math.min(maxSpan, Math.round(span * factor)))
        if (newSpan === span) return prev
        // Keep the data point under the cursor anchored: its index stays
        // at the same fractional x-position within the visible window.
        const anchorIdx = prev.startIndex + span * ratio
        let startIndex = Math.round(anchorIdx - newSpan * ratio)
        startIndex = Math.max(0, Math.min(maxSpan - newSpan, startIndex))
        const endIndex = startIndex + newSpan
        return { startIndex, endIndex }
      })
    }

    el.addEventListener("wheel", handler, { passive: false })
    return () => el.removeEventListener("wheel", handler)
  }, [data.length])

  const peak = data.reduce(
    (p, c) => (c.value > p.value ? c : p),
    data[0] ?? { name: "", value: 0 },
  )

  return (
    <div
      ref={containerRef}
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

          {/* hover-revealed range slider (the zoom). Kept mounted so the
              brush's internal start/end indices survive across hovers.
              fill gives it a subtle tinted background so the user
              actually notices "drag me to zoom" when it fades in. */}
          <Brush
            dataKey="name"
            height={22}
            stroke="var(--color-count)"
            fill="rgba(0, 172, 193, 0.08)"
            travellerWidth={8}
            startIndex={range.startIndex}
            endIndex={range.endIndex}
            onChange={(r) => {
              if (
                typeof r?.startIndex === "number" &&
                typeof r?.endIndex === "number" &&
                (r.startIndex !== range.startIndex ||
                  r.endIndex !== range.endIndex)
              ) {
                setRange({ startIndex: r.startIndex, endIndex: r.endIndex })
              }
            }}
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
