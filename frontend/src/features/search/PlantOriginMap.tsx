import { useEffect, useMemo, useState } from "react"
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  ZoomableGroup,
} from "react-simple-maps"
import { Skeleton } from "@/components/ui/skeleton"
import {
  buildCentroidMap,
  lookupCentroid,
  type Lnglat,
} from "@/lib/mapCentroids"

/**
 * Plant Origin Heatmap — react-simple-maps replacement for the previous
 * ECharts effectScatter geo. Matches the ECharts visual closely:
 *
 *   - Light "ghost map" world fill (#FFFFFF) with light teal borders
 *     (#D8E8EE), darker on hover (#9DE4EF).
 *   - Cyan pulsing markers (#06B6D4) sized by sqrt(value / max), with
 *     a stroke-only ring that expands 3.3x over 3.3s (matches the
 *     ECharts ``rippleEffect: { brushType: 'stroke', scale: 3.3,
 *     period: 3.3 }`` config exactly via pure CSS).
 *   - Per-marker animation-delay so the rings don't pulse in lockstep.
 *   - Custom hover tooltip positioned at the cursor. Uses shadcn
 *     ``bg-popover`` / ``text-popover-foreground`` tokens so dark
 *     mode auto-flips with the theme.
 *   - Pan + drag via ``<ZoomableGroup>`` (parity with ECharts
 *     ``geo.roam: true``).
 *
 * The world.json topology comes from /public/world.json — the SAME
 * file ECharts was reading. No re-download, same country names.
 *
 * Projection: ``geoNaturalEarth1`` — the rounded earth-globe look
 * common in academic data viz. Closest visual match to ECharts'
 * default world projection.
 */

interface Props {
  data: { name: string; value: number }[]
  onCountryClick?: (name: string) => void
  height?: number
}

interface Hovered {
  name: string
  value: number
  x: number
  y: number
}

// Type narrowing — react-simple-maps' generic geographies don't carry
// a property schema. Treat the rsmKey + properties.name optionally.
interface GeoFeature {
  rsmKey: string
  properties: { name?: string }
}

export function PlantOriginMap({ data, onCountryClick, height = 520 }: Props) {
  const [worldGeo, setWorldGeo] = useState<unknown>(null)
  const [centroids, setCentroids] = useState<Map<string, Lnglat>>(
    () => new Map(),
  )
  const [hovered, setHovered] = useState<Hovered | null>(null)

  // Fetch the world topology once on mount.
  useEffect(() => {
    let cancelled = false
    fetch("/world.json")
      .then((r) => r.json())
      .then((world) => {
        if (cancelled) return
        setWorldGeo(world)
        setCentroids(buildCentroidMap(world))
      })
      .catch(() => {
        if (!cancelled) setWorldGeo({})
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Resolve each data entry to a [lng, lat] + a base radius proportional
  // to ``sqrt(value / max)``. The sqrt scaling keeps small + large
  // values both visible — linear scaling would hide low-count countries.
  const markers = useMemo(() => {
    if (!data.length) return []
    const max = data.reduce((m, d) => Math.max(m, d.value), 1)
    return data
      .map((d) => {
        const coords = lookupCentroid(d.name, centroids)
        if (!coords) return null
        const baseR = Math.max(3, Math.sqrt(d.value / max) * 12)
        return { name: d.name, value: d.value, coords, baseR }
      })
      .filter(
        (m): m is { name: string; value: number; coords: Lnglat; baseR: number } =>
          m !== null,
      )
  }, [data, centroids])

  // Loading state — Skeleton matches the original chart height so the
  // dashboard layout doesn't shift when the map arrives.
  if (!worldGeo) {
    return <Skeleton className="rounded-2xl" style={{ height }} />
  }

  return (
    <div
      className="relative w-full overflow-hidden rounded-2xl"
      style={{ height, background: "#EDF5F8" }}
    >
      <ComposableMap
        projection="geoNaturalEarth1"
        projectionConfig={{ scale: 155 }}
        style={{ width: "100%", height: "100%" }}
      >
        <ZoomableGroup zoom={1} center={[0, 20]} minZoom={1} maxZoom={5}>
          <Geographies geography={worldGeo as object}>
            {({ geographies }: { geographies: GeoFeature[] }) =>
              geographies.map((g) => (
                <Geography
                  key={g.rsmKey}
                  geography={g}
                  style={{
                    default: {
                      fill: "#FFFFFF",
                      stroke: "#D8E8EE",
                      strokeWidth: 0.6,
                      outline: "none",
                    },
                    hover: {
                      fill: "#EBF8FB",
                      stroke: "#9DE4EF",
                      strokeWidth: 0.8,
                      outline: "none",
                      cursor: onCountryClick ? "pointer" : "default",
                    },
                    pressed: {
                      fill: "#EBF8FB",
                      outline: "none",
                    },
                  }}
                  onClick={() => {
                    const name = g.properties?.name
                    if (name && onCountryClick) onCountryClick(name)
                  }}
                />
              ))
            }
          </Geographies>

          {markers.map((m, i) => (
            <Marker
              key={`${m.name}-${i}`}
              coordinates={m.coords}
              onMouseEnter={(e) =>
                setHovered({
                  name: m.name,
                  value: m.value,
                  x: e.clientX,
                  y: e.clientY,
                })
              }
              onMouseMove={(e) =>
                setHovered((prev) =>
                  prev ? { ...prev, x: e.clientX, y: e.clientY } : prev,
                )
              }
              onMouseLeave={() => setHovered(null)}
              onClick={() => onCountryClick?.(m.name)}
            >
              {/* Stroke-only pulse ring — staggered start per marker */}
              <circle
                r={m.baseR}
                fill="none"
                stroke="#06B6D4"
                strokeWidth={1.5}
                className="map-pulse cursor-pointer"
                style={{ animationDelay: `${(i % 7) * 0.15}s` }}
              />
              {/* Solid center dot */}
              <circle
                r={m.baseR * 0.45}
                fill="#06B6D4"
                opacity={0.92}
                className="cursor-pointer"
              />
            </Marker>
          ))}
        </ZoomableGroup>
      </ComposableMap>

      {/* Tooltip — fixed-position, follows cursor */}
      {hovered && (
        <div
          className="fixed z-50 pointer-events-none rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs text-popover-foreground shadow-md"
          style={{ left: hovered.x + 10, top: hovered.y + 10 }}
        >
          <div className="font-semibold" style={{ color: "#0E7490" }}>
            {hovered.name}
          </div>
          <div className="text-muted-foreground">
            {hovered.value.toLocaleString()} papers
          </div>
        </div>
      )}
    </div>
  )
}
