import { useEffect, useMemo, useState } from "react"
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  ZoomableGroup,
} from "react-simple-maps"
// topojson-client is a transitive dep of react-simple-maps — we use it
// here to decode the world-atlas TopoJSON to a GeoJSON FeatureCollection
// once, then reuse that for both <Geographies> and centroid extraction.
import { feature } from "topojson-client"
import type { Topology } from "topojson-specification"
import {
  buildCentroidMap,
  lookupCentroid,
  splitMultiCountry,
  type Centroid,
  type Lnglat,
} from "@/lib/mapCentroids"

// GeoJSON FeatureCollection we hand to <Geographies> after decoding.
type WorldGeo = { type: "FeatureCollection"; features: unknown[] }

/**
 * Plant Origin Heatmap — react-simple-maps replacement for the previous
 * ECharts effectScatter geo.
 *
 * Data source: the canonical d3 `world-atlas` countries-110m TopoJSON
 * (bundled at /world-topo.json). We fetch + decode it once in this
 * component, pass the resulting FeatureCollection to <Geographies>,
 * and reuse the same decoded data for centroid extraction.
 */

const WORLD_GEO_URL = "/world-topo.json"

interface Props {
  data: { name: string; value: number }[]
  onCountryClick?: (name: string) => void
  height?: number
}

interface MarkerHover {
  name: string
  value: number
  x: number
  y: number
}

interface CountryHover {
  name: string
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
  const [world, setWorld] = useState<WorldGeo | null>(null)
  const [centroids, setCentroids] = useState<Map<string, Centroid>>(
    () => new Map(),
  )
  const [markerHover, setMarkerHover] = useState<MarkerHover | null>(null)
  const [countryHover, setCountryHover] = useState<CountryHover | null>(null)

  // Marker takes precedence over country if both are active (marker
  // sits on top of the country shape in z-order anyway).
  const tip: MarkerHover | CountryHover | null = markerHover ?? countryHover
  const tipHasValue = tip != null && "value" in tip

  // Fetch the world-atlas TopoJSON, decode it to a GeoJSON
  // FeatureCollection, then hand the SAME object to <Geographies>
  // (d3-geo path projection) and buildCentroidMap (marker placement).
  useEffect(() => {
    let cancelled = false
    fetch(WORLD_GEO_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`world-topo.json HTTP ${r.status}`)
        return r.json()
      })
      .then((topo: Topology) => {
        if (cancelled) return
        const fc = feature(topo, topo.objects.countries) as unknown as WorldGeo
        setWorld(fc)
        setCentroids(buildCentroidMap(fc))
      })
      .catch((err) => {
        console.error("[PlantOriginMap] failed to load world geo:", err)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const markers = useMemo(() => {
    if (!data.length || centroids.size === 0) return []

    // Aggregate by canonical world-atlas country. Multi-country mentions
    // like "Italy And France" or "Guinea, Uganda And Sudan" are split
    // and each component country gets the paper's contribution credited
    // (full attribution to each — the paper IS about both/all). Direct
    // lookup is tried first so single-country names containing " And " /
    // commas (Antigua and Barbuda, Trinidad and Tobago, etc.) keep their
    // own aliased centroid instead of getting split.
    const byCountry = new Map<
      string,
      { name: string; value: number; coords: Lnglat }
    >()

    const contribute = (rawName: string, value: number): boolean => {
      const c = lookupCentroid(rawName, centroids)
      if (!c) return false
      const existing = byCountry.get(c.name)
      if (existing) existing.value += value
      else byCountry.set(c.name, { name: c.name, value, coords: c.coords })
      return true
    }

    for (const d of data) {
      if (contribute(d.name, d.value)) continue
      const parts = splitMultiCountry(d.name)
      if (parts.length === 1) continue
      for (const part of parts) contribute(part, d.value)
    }

    const aggregated = Array.from(byCountry.values())
    const max = aggregated.reduce((m, d) => Math.max(m, d.value), 1)
    return aggregated.map(({ name, value, coords }) => ({
      name,
      value,
      coords,
      baseR: Math.max(3, Math.sqrt(value / max) * 12),
    }))
  }, [data, centroids])

  return (
    <div
      className="relative w-full overflow-hidden rounded-2xl"
      style={{ height, background: "#EDF5F8" }}
    >
      <ComposableMap
        projection="geoNaturalEarth1"
        projectionConfig={{ scale: 175, center: [0, 20] }}
        style={{ width: "100%", height: "100%" }}
      >
        <ZoomableGroup zoom={1} minZoom={1} maxZoom={5}>
          {world && (
            <Geographies geography={world}>
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
                    onMouseEnter={(e) => {
                      const name = g.properties?.name
                      if (name) {
                        setCountryHover({ name, x: e.clientX, y: e.clientY })
                      }
                    }}
                    onMouseMove={(e) =>
                      setCountryHover((prev) =>
                        prev ? { ...prev, x: e.clientX, y: e.clientY } : prev,
                      )
                    }
                    onMouseLeave={() => setCountryHover(null)}
                    onClick={() => {
                      const name = g.properties?.name
                      if (name && onCountryClick) onCountryClick(name)
                    }}
                  />
                ))
              }
            </Geographies>
          )}

          {markers.map((m, i) => (
            <Marker
              key={`${m.name}-${i}`}
              coordinates={m.coords}
              onMouseEnter={(e) =>
                setMarkerHover({
                  name: m.name,
                  value: m.value,
                  x: e.clientX,
                  y: e.clientY,
                })
              }
              onMouseMove={(e) =>
                setMarkerHover((prev) =>
                  prev ? { ...prev, x: e.clientX, y: e.clientY } : prev,
                )
              }
              onMouseLeave={() => setMarkerHover(null)}
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

      {/* Tooltip — same fixed-position floating chip used for both
          country hovers (name only) and marker hovers (name + count).
          Marker hover wins if both happen at the same coords. */}
      {tip && (
        <div
          className="fixed z-50 pointer-events-none rounded-lg border border-border/60 bg-popover/95 backdrop-blur-sm px-3 py-2 text-xs text-popover-foreground shadow-lg ring-1 ring-black/5"
          style={{ left: tip.x + 14, top: tip.y + 14 }}
        >
          <div className="flex items-center gap-1.5 font-semibold tracking-tight">
            {tipHasValue && (
              <span
                aria-hidden
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: "#06B6D4" }}
              />
            )}
            <span style={{ color: "#0E7490" }}>{tip.name}</span>
          </div>
          {tipHasValue && (
            <div className="mt-0.5 text-muted-foreground tabular-nums">
              {(tip as MarkerHover).value.toLocaleString()} papers
            </div>
          )}
        </div>
      )}
    </div>
  )
}
