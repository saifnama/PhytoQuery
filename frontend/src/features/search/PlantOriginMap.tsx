import { useEffect, useMemo, useState, useRef } from "react"
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
 * Data source: the canonical d3 `world-atlas` countries-50m TopoJSON
 * (bundled at /world-topo-50m.json — 241 countries / territories). We
 * fetch + decode it once in this component, pass the resulting
 * FeatureCollection to <Geographies>, and reuse the same decoded data
 * for centroid extraction.
 */

const WORLD_GEO_URL = "/world-topo-50m.json"

interface Props {
  data: { name: string; value: number }[]
  onCountryClick?: (name: string) => void
  height?: number
}

interface HoverData {
  name: string
  value?: number
}

// react-simple-maps' generic geographies don't carry a property
// schema; we just need rsmKey + an optional properties.name.
interface GeoFeature {
  rsmKey: string
  properties: { name?: string }
}

export function PlantOriginMap({ data, onCountryClick, height = 520 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const tooltipRef = useRef<HTMLDivElement | null>(null)
  const currentHoverRef = useRef<string | null>(null)
  const leaveTimerRef = useRef<number | null>(null)
  const [world, setWorld] = useState<WorldGeo | null>(null)
  const [centroids, setCentroids] = useState<Map<string, Centroid>>(
    () => new Map(),
  )
  const [hoverData, setHoverData] = useState<HoverData | null>(null)

  // Fetch the world-atlas TopoJSON, decode it to a GeoJSON
  // FeatureCollection, then hand the SAME object to <Geographies>
  // (d3-geo path projection) and buildCentroidMap (marker placement).
  useEffect(() => {
    let cancelled = false
    fetch(WORLD_GEO_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`world-topo-50m.json HTTP ${r.status}`)
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

  // Map of lowercase country name -> total papers for instant O(1) lookup on hover
  const countryValueMap = useMemo(() => {
    const map = new Map<string, number>()
    for (const m of markers) {
      map.set(m.name.toLowerCase(), m.value)
    }
    for (const d of data) {
      const existing = map.get(d.name.toLowerCase()) || 0
      map.set(d.name.toLowerCase(), Math.max(existing, d.value))
    }
    return map
  }, [markers, data])

  const getCountryPapers = (name: string): number | undefined => {
    const lower = name.toLowerCase()
    const direct = countryValueMap.get(lower)
    if (direct !== undefined) return direct
    const c = lookupCentroid(name, centroids)
    if (c) {
      return countryValueMap.get(c.name.toLowerCase())
    }
    return undefined
  }

  // Direct DOM hardware-accelerated tooltip positioning (0ms latency, no React SVG diffing)
  const updateTooltipPos = (clientX: number, clientY: number) => {
    if (!containerRef.current || !tooltipRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = clientX - rect.left + 14
    const y = clientY - rect.top + 14
    const maxX = rect.width - (tooltipRef.current.offsetWidth || 160) - 8
    const maxY = rect.height - (tooltipRef.current.offsetHeight || 60) - 8
    const clampedX = Math.max(8, Math.min(x, maxX))
    const clampedY = Math.max(8, Math.min(y, maxY))
    tooltipRef.current.style.transform = `translate3d(${clampedX}px, ${clampedY}px, 0)`
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden rounded-2xl"
      style={{ height, background: "#EDF5F8" }}
      onMouseMove={(e) => updateTooltipPos(e.clientX, e.clientY)}
      onMouseLeave={() => {
        if (leaveTimerRef.current) clearTimeout(leaveTimerRef.current)
        currentHoverRef.current = null
        setHoverData(null)
      }}
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
                      if (!name) return
                      if (leaveTimerRef.current) clearTimeout(leaveTimerRef.current)
                      updateTooltipPos(e.clientX, e.clientY)
                      if (currentHoverRef.current !== name) {
                        currentHoverRef.current = name
                        setHoverData({
                          name,
                          value: getCountryPapers(name),
                        })
                      }
                    }}
                    onMouseLeave={() => {
                      const name = g.properties?.name
                      if (leaveTimerRef.current) clearTimeout(leaveTimerRef.current)
                      leaveTimerRef.current = window.setTimeout(() => {
                        if (currentHoverRef.current === name) {
                          currentHoverRef.current = null
                          setHoverData(null)
                        }
                      }, 25)
                    }}
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
              onMouseEnter={(e) => {
                if (leaveTimerRef.current) clearTimeout(leaveTimerRef.current)
                updateTooltipPos(e.clientX, e.clientY)
                if (currentHoverRef.current !== m.name) {
                  currentHoverRef.current = m.name
                  setHoverData({
                    name: m.name,
                    value: m.value,
                  })
                }
              }}
              onMouseLeave={() => {
                const name = m.name
                if (leaveTimerRef.current) clearTimeout(leaveTimerRef.current)
                leaveTimerRef.current = window.setTimeout(() => {
                  if (currentHoverRef.current === name) {
                    currentHoverRef.current = null
                    setHoverData(null)
                  }
                }, 25)
              }}
              onClick={() => onCountryClick?.(m.name)}
            >
              {/* Three concentric ripple rings. pointer-events-none prevents hover jitter. */}
              {[0, 0.7, 1.4].map((beginSec, k) => (
                <circle
                  key={k}
                  cx={0}
                  cy={0}
                  r={m.baseR * 0.4}
                  fill="none"
                  stroke="#06B6D4"
                  strokeWidth={1.6}
                  vectorEffect="non-scaling-stroke"
                  className="pointer-events-none"
                >
                  <animate
                    attributeName="r"
                    values={`${m.baseR * 0.4}; ${m.baseR * 2.8}`}
                    keyTimes="0; 1"
                    dur="2.1s"
                    begin={`${beginSec}s`}
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="opacity"
                    values="0.7; 0.35; 0"
                    keyTimes="0; 0.6; 1"
                    dur="2.1s"
                    begin={`${beginSec}s`}
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="stroke-width"
                    values="1.8; 1.2; 0.4"
                    keyTimes="0; 0.6; 1"
                    dur="2.1s"
                    begin={`${beginSec}s`}
                    repeatCount="indefinite"
                  />
                </circle>
              ))}

              {/* Larger hit area for easy hover targeting */}
              <circle
                cx={0}
                cy={0}
                r={Math.max(12, m.baseR * 1.5)}
                fill="transparent"
                className="cursor-pointer"
              />

              {/* Solid center dot. */}
              <circle
                cx={0}
                cy={0}
                r={m.baseR * 0.55}
                fill="#06B6D4"
                opacity={0.95}
                className="cursor-pointer pointer-events-none"
              />
            </Marker>
          ))}
        </ZoomableGroup>
      </ComposableMap>

      {/* Tooltip — direct GPU translate3d hardware accelerated with zero React SVG re-renders */}
      <div
        ref={tooltipRef}
        className={`
          absolute top-0 left-0 z-50 pointer-events-none rounded-lg
          border border-border/60 bg-popover/95 backdrop-blur-sm
          px-3 py-2 text-xs text-popover-foreground shadow-lg ring-1 ring-black/5
          will-change-transform
          ${hoverData ? "opacity-100" : "opacity-0 pointer-events-none"}
        `}
        style={{
          transform: "translate3d(-9999px, -9999px, 0)",
          transition: "opacity 60ms ease-out",
        }}
      >
        {hoverData && (
          <>
            <div className="flex items-center gap-1.5 font-semibold tracking-tight">
              {hoverData.value != null && hoverData.value > 0 && (
                <span
                  aria-hidden
                  className="inline-block h-2 w-2 rounded-full shrink-0"
                  style={{ background: "#06B6D4" }}
                />
              )}
              <span style={{ color: "#0E7490" }} className="truncate max-w-[200px]">
                {hoverData.name}
              </span>
            </div>
            {hoverData.value != null && hoverData.value > 0 && (
              <div className="mt-0.5 text-muted-foreground tabular-nums font-medium">
                {hoverData.value.toLocaleString()} papers
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
