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
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
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
      className="flex overflow-hidden rounded-2xl border"
      style={{ height, borderColor: "#C8F1F8" }}
    >
      {/* Left pane — shadcn Card with full Header/Content/Footer composition.
          justify-between spreads the three sections across the pane. */}
      <Card
        className="flex-1 rounded-none ring-0 gap-0 justify-between py-6"
        style={{ background: "#F2FBFC" }}
      >
        <CardHeader>
          <CardTitle
            className="text-sm font-medium line-clamp-2 leading-tight"
            style={{ color: "#1A5F6B" }}
          >
            {dominant.name}
          </CardTitle>
        </CardHeader>

        <CardContent>
          <div className="flex items-baseline gap-2">
            <span
              className="text-6xl font-semibold tabular-nums leading-none"
              style={{ color: "#2AACBF" }}
            >
              {dominant.value.toLocaleString()}
            </span>
            <span className="text-sm" style={{ color: "#5BBCC8" }}>
              papers
            </span>
          </div>
        </CardContent>

        <CardFooter className="flex-col items-stretch gap-2">
          <div className="flex w-full items-baseline justify-between">
            <span
              className="text-2xl font-medium tabular-nums"
              style={{ color: "#2AACBF" }}
            >
              {pct}%
            </span>
            <span className="text-xs" style={{ color: "#5BBCC8" }}>
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
        </CardFooter>
      </Card>

      {/* Inter-pane divider — shadcn Separator. The default --border
          token would be too dark on this background, so it's
          explicitly tinted with the Coastal border colour. */}
      <Separator
        orientation="vertical"
        className="h-auto bg-[#C8F1F8]"
      />

      {/* Right pane — shadcn Card hosting the bar chart. */}
      <Card
        className="w-[240px] rounded-none ring-0 gap-0 py-3"
        style={{ background: "#FBFEFE" }}
      >
        <CardContent className="h-full px-3">
          <JournalBarsChart data={ranked} onBarClick={onJournalClick} />
        </CardContent>
      </Card>
    </div>
  )
}

export default JournalDistributionWidget
