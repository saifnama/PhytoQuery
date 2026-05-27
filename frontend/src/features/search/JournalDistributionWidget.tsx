/**
 * Journal Distribution Widget — shadcn-native rewrite.
 *
 * Two panes:
 *   Left  — dominant-journal hero in plain JSX + Tailwind. The
 *           previous implementation abused ECharts' ``graphic`` API
 *           as an SVG/canvas drawing primitive even though no chart
 *           series was being plotted. JSX is the right tool here:
 *           less code, theme-aware (no manual dark-mode branches),
 *           a11y-friendly out of the box.
 *   Right — ranked horizontal bar chart via ``<JournalBarsChart />``
 *           (shadcn "Bar Chart - Horizontal" pattern, Recharts).
 *
 * The component's public API is unchanged so Dashboard.tsx needs no
 * call-site update.
 */

import React from "react"
import { JournalBarsChart } from "./JournalBarsChart"

interface JournalEntry {
  name: string
  value: number
}

interface Props {
  journals: JournalEntry[] // sorted desc by paper count, dominant first
  totalPapers: number // for percentage calc
  height?: number // pane height in px (default 260)
  onJournalClick?: (name: string) => void // fires on right-pane bar click
}

const JournalDistributionWidget: React.FC<Props> = ({
  journals,
  totalPapers,
  height = 260,
  onJournalClick,
}) => {
  if (!journals.length) {
    return (
      <div
        className="flex items-center justify-center text-xs text-muted-foreground"
        style={{ height }}
      >
        No journal data
      </div>
    )
  }

  const dominant = journals[0]
  const ranked = journals.slice(1, 10)
  const pct =
    totalPapers > 0 ? Math.round((dominant.value / totalPapers) * 100) : 0

  return (
    <div
      className="grid overflow-hidden rounded-2xl border border-border"
      style={{ gridTemplateColumns: "1fr 1px 240px", height }}
    >
      {/* Left pane — hero stat, plain JSX */}
      <div className="flex flex-col justify-between p-6 bg-muted/30">
        <div className="text-sm font-medium text-foreground line-clamp-2 leading-tight">
          {dominant.name}
        </div>

        <div className="flex items-baseline gap-2">
          <span
            className="text-6xl font-semibold tabular-nums text-foreground"
            style={{ color: "var(--chart-1)" }}
          >
            {dominant.value.toLocaleString()}
          </span>
          <span className="text-sm text-muted-foreground">papers</span>
        </div>

        <div className="space-y-2">
          <div className="flex items-baseline justify-between">
            <span
              className="text-2xl font-medium tabular-nums"
              style={{ color: "var(--chart-1)" }}
            >
              {pct}%
            </span>
            <span className="text-xs text-muted-foreground">of corpus</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${pct}%`,
                backgroundColor: "var(--chart-1)",
              }}
            />
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="bg-border" />

      {/* Right pane — ranked horizontal bars */}
      <div className="p-3 overflow-hidden bg-card">
        <JournalBarsChart data={ranked} onBarClick={onJournalClick} />
      </div>
    </div>
  )
}

export default JournalDistributionWidget
