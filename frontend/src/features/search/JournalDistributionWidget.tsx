/**
 * Journal Distribution Widget — shadcn-native composition.
 *
 * Both panes are shadcn ``<Card>`` instances; the dividing rule is a
 * shadcn ``<Separator orientation="vertical">``. The outer wrapper is
 * a layout-only flex container that gives the unified rounded border
 * (no nested Card rings).
 *
 * Colors are the original Coastal palette hex values (preserved at the
 * user's request — these specific teals don't map cleanly to shadcn
 * semantic tokens):
 *   - #1A5F6B  hero title
 *   - #2AACBF  big-number + percentage
 *   - #5BBCC8  caption text
 *   - #A0E4F1  progress-bar fill
 *   - #F2FBFC  left-pane background
 *   - #FBFEFE  right-pane background
 *   - #C8F1F8  outer border + inter-pane separator
 *
 * Card defaults are overridden in two places: ``rounded-none ring-0``
 * (the outer div already provides the rounded border, so each inner
 * Card's own ring would double up), and ``gap-0`` (the hero pane uses
 * ``justify-between`` to distribute its three sections across the full
 * pane height instead of shadcn's default 24px gaps).
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
  const ranked = journals.slice(1, 9)
  const pct =
    totalPapers > 0 ? Math.round((dominant.value / totalPapers) * 100) : 0

  return (
    <div
      className="flex items-stretch gap-4 w-full"
      style={{ height }}
    >
      {/* Left pane — blue card (equal 50% split, completely borderless) */}
      <div
        className="flex-1 min-w-0 rounded-2xl flex flex-col justify-between py-6 px-6"
        style={{
          fontFamily: "var(--font-google-sans)",
          background: "#F2FBFC",
          border: "none",
          outline: "none",
          boxShadow: "none",
        }}
      >
        <div>
          <div
            className="text-[17.5px] font-semibold line-clamp-2 leading-snug tracking-tight"
            style={{ color: "#2AACBF", fontFamily: "var(--font-google-sans)" }}
          >
            {dominant.name}
          </div>
        </div>

        <div>
          <div className="flex items-baseline gap-2">
            <span
              className="text-6xl font-semibold tabular-nums leading-none"
              style={{ color: "#2AACBF", fontFamily: "var(--font-google-sans)" }}
            >
              {dominant.value.toLocaleString()}
            </span>
            <span className="text-sm font-medium" style={{ color: "#5BBCC8" }}>
              papers
            </span>
          </div>
        </div>

        <div className="flex flex-col items-stretch gap-2">
          <div className="flex w-full items-baseline justify-between">
            <span
              className="text-2xl font-medium tabular-nums"
              style={{ color: "#2AACBF", fontFamily: "var(--font-google-sans)" }}
            >
              {pct}%
            </span>
            <span className="text-xs font-medium" style={{ color: "#5BBCC8" }}>
              of corpus
            </span>
          </div>
          <div
            className="h-1.5 w-full overflow-hidden rounded-full"
            style={{ backgroundColor: "rgba(160,228,241,0.3)" }}
          >
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${pct}%`,
                backgroundColor: "#A0E4F1",
              }}
            />
          </div>
        </div>
      </div>

      {/* Right pane — borderless chart */}
      <div className="flex-1 min-w-0 h-full">
        <JournalBarsChart data={ranked} onBarClick={onJournalClick} />
      </div>
    </div>
  )
}

export default JournalDistributionWidget
