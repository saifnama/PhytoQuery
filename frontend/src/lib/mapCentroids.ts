/**
 * Country-centroid utilities for the dashboard's geographic map.
 *
 * Consumes a GeoJSON ``FeatureCollection`` (e.g. the one in
 * ``/public/world.json``) and returns a name → [lng, lat] map that
 * the map component uses to position markers. The largest polygon
 * of each country wins so multi-island countries (Indonesia, Japan,
 * UK …) get a centroid on the main landmass rather than drifting
 * offshore.
 *
 * GeoJSON in the wild is messy — empty rings, NaN coords,
 * GeometryCollection wrappers, missing properties. Every helper
 * here treats malformed input as "skip, don't throw" so one bad
 * feature can never poison the centroid map.
 */

export type Lnglat = [number, number]

// Aliases map NER / Excel variants to the EXACT canonical names used
// in /public/world.json (short English names — "Russia", "Turkey",
// "Iran", "Vietnam", "Korea", "Czech Rep.", "Lao PDR",
// "Dem. Rep. Korea", "Dem. Rep. Congo").
export const COUNTRY_ALIASES: Record<string, string> = {
  usa: "United States",
  "u.s.a.": "United States",
  "u.s.": "United States",
  us: "United States",
  "united states of america": "United States",
  america: "United States",
  uk: "United Kingdom",
  "u.k.": "United Kingdom",
  britain: "United Kingdom",
  "great britain": "United Kingdom",
  "russian federation": "Russia",
  türkiye: "Turkey",
  turkiye: "Turkey",
  "iran (islamic republic of)": "Iran",
  "syrian arab republic": "Syria",
  "viet nam": "Vietnam",
  "venezuela (bolivarian republic of)": "Venezuela",
  "bolivia (plurinational state of)": "Bolivia",
  "united republic of tanzania": "Tanzania",
  "republic of moldova": "Moldova",
  "north macedonia": "Macedonia",
  "south korea": "Korea",
  "republic of korea": "Korea",
  "north korea": "Dem. Rep. Korea",
  "korea, democratic people's republic of": "Dem. Rep. Korea",
  "czech republic": "Czech Rep.",
  czechia: "Czech Rep.",
  laos: "Lao PDR",
  "lao people's democratic republic": "Lao PDR",
  drc: "Dem. Rep. Congo",
  "democratic republic of congo": "Dem. Rep. Congo",
  "democratic republic of the congo": "Dem. Rep. Congo",
  "ivory coast": "Côte d'Ivoire",
  "cote d'ivoire": "Côte d'Ivoire",
  "cote d ivoire": "Côte d'Ivoire",
}

function isLnglatPoint(p: unknown): p is Lnglat {
  return (
    Array.isArray(p) &&
    p.length >= 2 &&
    typeof p[0] === "number" &&
    Number.isFinite(p[0]) &&
    typeof p[1] === "number" &&
    Number.isFinite(p[1])
  )
}

function isRing(r: unknown): r is Lnglat[] {
  return Array.isArray(r) && r.length > 0 && r.every(isLnglatPoint)
}

function ringArea(ring: Lnglat[]): number {
  if (ring.length < 3) return 0
  let a = 0
  for (let i = 0; i < ring.length - 1; i++) {
    a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
  }
  return Math.abs(a / 2)
}

function ringBboxCenter(ring: Lnglat[]): Lnglat | null {
  let xMin = Infinity,
    yMin = Infinity,
    xMax = -Infinity,
    yMax = -Infinity
  for (const [x, y] of ring) {
    if (x < xMin) xMin = x
    if (x > xMax) xMax = x
    if (y < yMin) yMin = y
    if (y > yMax) yMax = y
  }
  if (!Number.isFinite(xMin) || !Number.isFinite(yMin)) return null
  const cx = (xMin + xMax) / 2
  const cy = (yMin + yMax) / 2
  return Number.isFinite(cx) && Number.isFinite(cy) ? [cx, cy] : null
}

// Walks any GeoJSON geometry — Polygon, MultiPolygon, or
// GeometryCollection — and returns every valid outer ring it finds.
function collectOuterRings(geom: unknown): Lnglat[][] {
  const rings: Lnglat[][] = []
  if (!geom || typeof geom !== "object") return rings
  const g = geom as { type?: string; coordinates?: unknown; geometries?: unknown[] }
  if (g.type === "Polygon") {
    const ring = (g.coordinates as unknown[] | undefined)?.[0]
    if (isRing(ring)) rings.push(ring)
  } else if (g.type === "MultiPolygon") {
    for (const poly of (g.coordinates as unknown[] | undefined) ?? []) {
      const ring = (poly as unknown[] | undefined)?.[0]
      if (isRing(ring)) rings.push(ring)
    }
  } else if (g.type === "GeometryCollection") {
    for (const sub of g.geometries ?? []) {
      rings.push(...collectOuterRings(sub))
    }
  }
  return rings
}

function featureCentroid(feature: unknown): Lnglat | null {
  const f = feature as { geometry?: unknown }
  const rings = collectOuterRings(f?.geometry)
  if (rings.length === 0) return null
  // Largest polygon (by area) wins — keeps centroids on the main
  // landmass for multi-island countries.
  let largest = rings[0]
  let maxArea = ringArea(largest)
  for (let i = 1; i < rings.length; i++) {
    const a = ringArea(rings[i])
    if (a > maxArea) {
      maxArea = a
      largest = rings[i]
    }
  }
  return ringBboxCenter(largest)
}

export function buildCentroidMap(world: unknown): Map<string, Lnglat> {
  const map = new Map<string, Lnglat>()
  const features = (world as { features?: unknown[] } | null)?.features ?? []
  if (!Array.isArray(features)) return map
  let failed = 0
  for (const f of features) {
    try {
      const name = (f as { properties?: { name?: unknown } } | null)?.properties?.name
      if (typeof name !== "string" || !name.trim()) continue
      const c = featureCentroid(f)
      if (c) map.set(name.toLowerCase(), c)
    } catch {
      failed += 1
      // one bad feature must not break the rest of the map
    }
  }
  if (failed > 0) {
    console.warn(`[mapCentroids] skipped ${failed} malformed feature(s)`)
  }
  return map
}

export function lookupCentroid(
  rawName: string,
  centroids: Map<string, Lnglat>,
): Lnglat | null {
  const key = rawName.trim().toLowerCase()
  const direct = centroids.get(key)
  if (direct) return direct
  const aliased = COUNTRY_ALIASES[key]
  return aliased ? (centroids.get(aliased.toLowerCase()) ?? null) : null
}
