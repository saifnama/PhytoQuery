import { useMemo, useState } from "react"
import { Cell, Label, Pie, PieChart } from "recharts"
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { getEntityColor } from "@/lib/entityColors"

/**
 * Entity Donut — shadcn "Pie Chart - Donut with Text" pattern, with
 * semantic per-entity colours from ENTITY_COLORS (chemical=blue,
 * species=green, location=cyan, etc.). This matches the previous
 * ECharts dashboard look exactly — each entity TYPE has a stable,
 * meaningful colour rather than cycling positionally through
 * ``--chart-1`` … ``--chart-5``.
 *
 * Center text behaviour:
 *   - default: shows the total entity count + "entities" label
 *   - on slice hover: shows the active slice's name + value
 *
 * ``paddingAngle`` provides visual separation between adjacent slices.
 */

interface Props {
  data: { name: string; value: number }[]
  onSliceClick?: (name: string) => void
}

// Stable key derivation for chartConfig — strip non-alphanumerics, lowercase,
// collapse runs. Mirrors shadcn's "snake_lower" idiom for keys derived from
// human-readable labels.
function entityKey(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
}

export function EntityDonutChart({ data, onSliceClick }: Props) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)

  const total = useMemo(
    () => data.reduce((s, d) => s + d.value, 0),
    [data],
  )

  // chartConfig is rebuilt from data so each entity name maps to its
  // SEMANTIC colour from ENTITY_COLORS (chemical=blue, species=green,
  // location=cyan, etc.). ChartContainer exposes these as
  // `--color-<key>` automatically, so tooltips + legend stay consistent
  // with the slice colour without any extra plumbing.
  const chartConfig = useMemo<ChartConfig>(() => {
    const config: ChartConfig = {
      value: { label: "Entities" },
    }
    data.forEach((d) => {
      config[entityKey(d.name)] = {
        label: d.name,
        color: getEntityColor(d.name).hex,
      }
    })
    return config
  }, [data])

  const dataWithFill = useMemo(
    () =>
      data.map((d) => ({
        ...d,
        key: entityKey(d.name),
        fill: getEntityColor(d.name).hex,
      })),
    [data],
  )

  const active = activeIndex != null ? dataWithFill[activeIndex] : null
  const centerLine1 = active
    ? active.value.toLocaleString()
    : total.toLocaleString()
  const centerLine2 = active ? active.name : "entities"

  return (
    <ChartContainer
      config={chartConfig}
      className="mx-auto aspect-square h-[280px] w-full"
    >
      <PieChart>
        <ChartTooltip
          cursor={false}
          content={
            <ChartTooltipContent
              hideLabel
              nameKey="name"
              formatter={(value, _name, item) => {
                const datum = item.payload as { name: string; value: number }
                return (
                  <div className="flex w-full justify-between gap-4">
                    <span className="text-muted-foreground">{datum.name}</span>
                    <span className="font-mono font-medium tabular-nums">
                      {Number(value).toLocaleString()}
                    </span>
                  </div>
                )
              }}
            />
          }
        />
        <Pie
          data={dataWithFill}
          dataKey="value"
          nameKey="name"
          innerRadius={70}
          outerRadius={110}
          strokeWidth={2}
          paddingAngle={2}
          onMouseEnter={(_, i) => setActiveIndex(i)}
          onMouseLeave={() => setActiveIndex(null)}
          onClick={(d: unknown) => {
            const payload = (d as { name?: string } | undefined)?.name
            if (payload && onSliceClick) onSliceClick(payload)
          }}
        >
          {dataWithFill.map((d) => (
            <Cell key={d.key} fill={d.fill} className="cursor-pointer" />
          ))}

          {/* shadcn "Donut with Text" center label — animates between
              total and the active slice's value/name */}
          <Label
            content={({ viewBox }) => {
              if (!viewBox || !("cx" in viewBox) || !("cy" in viewBox))
                return null
              const cx = viewBox.cx ?? 0
              const cy = viewBox.cy ?? 0
              return (
                <text
                  x={cx}
                  y={cy}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="select-none"
                >
                  <tspan
                    x={cx}
                    y={cy - 6}
                    className="fill-foreground text-3xl font-bold tabular-nums"
                  >
                    {centerLine1}
                  </tspan>
                  <tspan
                    x={cx}
                    y={cy + 18}
                    className="fill-muted-foreground text-xs"
                  >
                    {centerLine2}
                  </tspan>
                </text>
              )
            }}
          />
        </Pie>
      </PieChart>
    </ChartContainer>
  )
}
