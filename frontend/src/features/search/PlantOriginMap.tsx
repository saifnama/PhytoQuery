import { useEffect, useMemo, useState } from "react"
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  ZoomableGroup,
} from "react-simple-maps"
import {
  buildCentroidMap,
  lookupCentroid,
  type Lnglat,
} from "@/lib/mapCentroids"

/**
 * Plant Origin Heatmap — react-simple-maps replacement for the previous
 * ECharts effectScatter geo.
 *
 *   - Light "ghost map" world fill (#FFFFFF) with light teal borders
 *     (#D8E8EE), darker on hover (#9DE4EF).
 *   - Cyan markers (#06B6D4) sized by sqrt(value / max).
 *   - Ripple effect: **three concentric rings per marker**, each on
 *     the same 2.1s animation cycle but with -0s / -0.7s / -1.4s
 *     animation-delays so a new ring starts every 0.7s and three are
 *     always visible at different scales/opacities. That's the
 *     raindrop-on-water look — a single ring fading to zero left
 *     visible dead gaps between cycles.
 *   - ``vector-effect="non-scaling-stroke"`` keeps the ring 1.6px
 *     thick regardless of scale, matching the original ECharts
 *     ``rippleEffect: { brushType: 'stroke' }`` look.
 *
 * Geographies consume the world.json URL directly so react-simple-maps
 * handles the TopoJSON / GeoJSON resolution internally (passing a
 * pre-parsed object can fail when the format doesn't match RSM's
 * expectations exactly). We still fetch world.json ourselves once to
 * build the centroid map for marker positioning — RSM caches by URL
 * so the duplicate fetch is free.
 */

const WORLD_GEO_URL = "/world.json"

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

// react-simple-maps' generic geographies don't carry a property
// schema; we just need rsmKey + an optional properties.name.
interface GeoFeature {
  rsmKey: string
  properties: { name?: string }
}

export function PlantOriginMap({ data, onCountryClick, height = 520 }: Props) {
  const [centroids, setCentroids] = useState<Map<string, Lnglat>>(
    () => new Map(),
  )
  const [hovered, setHovered] = useState<Hovered | null>(null)

  // Fetch world.json once for centroid extraction. The Geographies
  // component does its own (cached) fetch via the URL prop, so there's
  // no double download in practice.
  useEffect(() => {
    let cancelled = false
    fetch(WORLD_GEO_URL)
      .then((r) => r.json())
      .then((world) => {
        if (!cancelled) setCentroids(buildCentroidMap(world))
      })
      .catch(() => {
        /* map keeps rendering without markers */
      })
    return () => {
      cancelled = true
    }
  }, [])

  const markers = useMemo(() => {
    if (!data.length || centroids.size === 0) return []
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

  return (
    <div
      className="relative w-full overflow-hidden rounded-2xl"
      style={{ height, background: "#EDF5F8" }}
    >
      <ComposableMap
        projection="geoNaturalEarth1"
        projectionConfig={{ scale: 175 }}
        style={{ width: "100%", height: "100%" }}
      >
        <ZoomableGroup zoom={1} center={[0, 20]} minZoom={1} maxZoom={5}>
          <Geographies geography={WORLD_GEO_URL}>
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
              {/* Three concentric ripple rings. Each runs the same 2.1s
                  cycle but with animation-delays of 0, -0.7s, -1.4s so
                  a new ring starts every 0.7s and three are always on
                  screen at different scales — the raindrop-on-water look.
                  vector-effect keeps the stroke a constant 1.6px as the
                  ring scales up. */}
              {[0, -0.7, -1.4].map((delay, k) => (
                <circle
                  key={k}
                  r={m.baseR}
                  fill="none"
                  stroke="#06B6D4"
                  strokeWidth={1.6}
                  vectorEffect="non-scaling-stroke"
                  className="map-ripple cursor-pointer"
                  style={{ animationDelay: `${delay}s` }}
                />
              ))}

              {/* Solid center dot. */}
              <circle
                r={m.baseR * 0.55}
                fill="#06B6D4"
                opacity={0.95}
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
