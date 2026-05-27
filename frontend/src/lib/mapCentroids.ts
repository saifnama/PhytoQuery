/**
 * Country-centroid utilities for the dashboard's geographic map.
 *
 * Consumes a GeoJSON ``FeatureCollection`` (the decoded form of
 * ``/public/world-topo.json`` — the d3 ``world-atlas`` countries-110m
 * TopoJSON) and returns a name → [lng, lat] map that the map component
 * uses to position markers. The largest polygon of each country wins so
 * multi-island countries (Indonesia, Japan, UK …) get a centroid on the
 * main landmass rather than drifting offshore.
 *
 * GeoJSON in the wild is messy — empty rings, NaN coords,
 * GeometryCollection wrappers, missing properties. Every helper
 * here treats malformed input as "skip, don't throw" so one bad
 * feature can never poison the centroid map.
 */

export type Lnglat = [number, number]

export interface Centroid {
  name: string // canonical display name (correctly cased, used in tooltip)
  coords: Lnglat
}

// Centroids for countries / dependencies absent from world-atlas-110m.
// Coords are [lng, lat] of the geographic center. A marker drops at
// the right spot even though world-atlas doesn't ship a country shape.
const HARDCODED_CENTROIDS: Record<string, Centroid> = {
  malta:                   { name: "Malta",                 coords: [14.4, 35.9] },
  mauritius:               { name: "Mauritius",             coords: [57.5, -20.2] },
  curaçao:                 { name: "Curaçao",               coords: [-69.0, 12.2] },
  curacao:                 { name: "Curaçao",               coords: [-69.0, 12.2] },
  martinique:              { name: "Martinique",            coords: [-61.0, 14.7] },
  "french polynesia":      { name: "French Polynesia",      coords: [-149.5, -17.5] },
  "isle of man":           { name: "Isle of Man",           coords: [-4.5, 54.2] },
  "antigua and barbuda":   { name: "Antigua and Barbuda",   coords: [-61.8, 17.1] },
  "são tomé and príncipe": { name: "São Tomé and Príncipe", coords: [6.6, 0.2] },
}

// Aliases map NER / Excel / ingest-time variants to the EXACT canonical
// names used in world-atlas countries-110m (plus the HARDCODED names
// above for territories world-atlas omits). world-atlas uses some
// idiosyncratic short forms — "United States of America" (NOT "United
// States"), "Czechia" (NOT "Czech Rep."), "Laos" (NOT "Lao PDR"),
// "Macedonia" (NOT "North Macedonia"), "Bosnia and Herz.",
// "Dominican Rep.", "Dem. Rep. Congo" — so the alias targets must match
// those strings byte-for-byte or the centroid lookup misses.
export const COUNTRY_ALIASES: Record<string, string> = {
  // United States
  usa: "United States of America",
  "u.s.a.": "United States of America",
  "u.s.": "United States of America",
  us: "United States of America",
  "united states": "United States of America",
  america: "United States of America",
  // United Kingdom
  uk: "United Kingdom",
  "u.k.": "United Kingdom",
  britain: "United Kingdom",
  "great britain": "United Kingdom",
  england: "United Kingdom",
  // Russia / Turkey / Iran / Syria / Vietnam
  "russian federation": "Russia",
  türkiye: "Turkey",
  turkiye: "Turkey",
  "iran (islamic republic of)": "Iran",
  "syrian arab republic": "Syria",
  "viet nam": "Vietnam",
  // South America
  "venezuela (bolivarian republic of)": "Venezuela",
  "venezuela bolivarian republic of": "Venezuela",
  "bolivia (plurinational state of)": "Bolivia",
  // Africa
  "united republic of tanzania": "Tanzania",
  morrocco: "Morocco",
  "burkina-faso": "Burkina Faso",
  // Eastern Europe
  "republic of moldova": "Moldova",
  "north macedonia": "Macedonia",
  macedoni: "Macedonia", // ingest typo
  hrvatska: "Croatia",
  deutschland: "Germany",
  slovensko: "Slovakia",
  bosnia: "Bosnia and Herz.",
  kosova: "Kosovo",
  // Korea (world-atlas keeps them split — default bare "Korea" to South)
  korea: "South Korea",
  "south korea": "South Korea",
  "republic of korea": "South Korea",
  "north korea": "North Korea",
  "korea, democratic people's republic of": "North Korea",
  // Czechia / Laos
  "czech republic": "Czechia",
  czechia: "Czechia",
  laos: "Laos",
  "lao people's democratic republic": "Laos",
  // DRC
  drc: "Dem. Rep. Congo",
  "democratic republic of congo": "Dem. Rep. Congo",
  "democratic republic of the congo": "Dem. Rep. Congo",
  // Ivory Coast
  "ivory coast": "Côte d'Ivoire",
  "cote d'ivoire": "Côte d'Ivoire",
  "cote d ivoire": "Côte d'Ivoire",
  // Asia
  "taiwan province of china": "Taiwan",
  // Caribbean
  "dominican republic": "Dominican Rep.",
  // Cities → owning country
  havana: "Cuba",
  // São Tomé encoding artifacts
  "s. tom� and pr�ncipe": "São Tomé and Príncipe",
  "s. tom?� and pr??ncipe": "São Tomé and Príncipe",
  "sao tome and principe": "São Tomé and Príncipe",
  "sao tome": "São Tomé and Príncipe",
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

export function buildCentroidMap(world: unknown): Map<string, Centroid> {
  const map = new Map<string, Centroid>()
  const features = (world as { features?: unknown[] } | null)?.features ?? []
  if (Array.isArray(features)) {
    let failed = 0
    for (const f of features) {
      try {
        const name = (f as { properties?: { name?: unknown } } | null)?.properties?.name
        if (typeof name !== "string" || !name.trim()) continue
        const coords = featureCentroid(f)
        if (coords) map.set(name.toLowerCase(), { name, coords })
      } catch {
        failed += 1
        // one bad feature must not break the rest of the map
      }
    }
    if (failed > 0) {
      console.warn(`[mapCentroids] skipped ${failed} malformed feature(s)`)
    }
  }
  // Merge hardcoded territories — world-atlas shapes win where they
  // exist; hardcoded entries only fill the gaps.
  for (const [key, val] of Object.entries(HARDCODED_CENTROIDS)) {
    if (!map.has(key)) map.set(key, val)
  }
  return map
}

export function lookupCentroid(
  rawName: string,
  centroids: Map<string, Centroid>,
): Centroid | null {
  const key = rawName.trim().toLowerCase()
  const direct = centroids.get(key)
  if (direct) return direct
  const aliased = COUNTRY_ALIASES[key]
  return aliased ? (centroids.get(aliased.toLowerCase()) ?? null) : null
}

// Split "Italy And France" / "Guinea, Uganda And Sudan" into components.
// Case-sensitive on " And " (capital A) so it ONLY fires on the DB's
// multi-country separator style — the lowercase " and " inside real
// country names ("Bosnia and Herz.", "Antigua and Barbuda", "Trinidad
// and Tobago", "Saint Vincent and the Grenadines") is preserved.
// Callers should still try the raw name through ``lookupCentroid``
// first; only fall back to splitting when direct lookup fails.
export function splitMultiCountry(rawName: string): string[] {
  const parts = rawName
    .split(/\s+And\s+|,\s+/)
    .map((s) => s.trim())
    .filter(Boolean)
  return parts.length > 1 ? parts : [rawName]
}
