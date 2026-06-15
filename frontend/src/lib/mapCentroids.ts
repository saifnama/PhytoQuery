/**
 * Country-centroid utilities for the dashboard's geographic map.
 *
 * Consumes a GeoJSON ``FeatureCollection`` (the decoded form of
 * ``/public/world-topo-50m.json`` — the d3 ``world-atlas`` countries-50m
 * TopoJSON, 241 countries / territories) and returns a name → centroid
 * map that the map component uses to position markers. The largest
 * polygon of each country wins so multi-island countries (Indonesia,
 * Japan, UK …) get a centroid on the main landmass rather than
 * drifting offshore. Antimeridian-crossing polygons (Russia, Fiji)
 * are unwrapped before averaging so their centroids don't collapse to
 * longitude 0.
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

// Aliases map NER / Excel / ingest-time variants to the EXACT canonical
// names used in world-atlas countries-50m. Every centroid is computed
// from the polygon geometry — no hardcoded coordinates anywhere. The
// 50m source covers 241 countries / territories (including the small
// island states that the smaller 110m omitted), so all centroids are
// algorithmic.
//
// world-atlas uses some idiosyncratic short forms — "United States of
// America" (NOT "United States"), "Czechia" (NOT "Czech Rep."), "Laos"
// (NOT "Lao PDR"), "Macedonia" (NOT "North Macedonia"), "Bosnia and
// Herz.", "Dominican Rep.", "Dem. Rep. Congo", "Fr. Polynesia",
// "Antigua and Barb.", "São Tomé and Principe" (no acute on Principe) —
// so the alias targets must match those strings byte-for-byte or the
// centroid lookup misses.
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
  "antigua and barbuda": "Antigua and Barb.",
  curacao: "Curaçao",
  // Oceania
  "french polynesia": "Fr. Polynesia",
  // Cities → owning country
  havana: "Cuba",
  // São Tomé encoding artifacts (50m uses "Principe" without acute)
  "s. tom� and pr�ncipe": "São Tomé and Principe",
  "s. tom?� and pr??ncipe": "São Tomé and Principe",
  "sao tome and principe": "São Tomé and Principe",
  "sao tome": "São Tomé and Principe",
  "são tomé and príncipe": "São Tomé and Principe",
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

  let cx: number
  if (xMax - xMin > 270) {
    // Polygon crosses the antimeridian (Russia is the classic case —
    // Chukotka sits east of the date line, the rest is west). A naïve
    // (xMin + xMax) / 2 averages -180 and +180 to 0 and drops the marker
    // in the North Sea. Shift every negative longitude by +360 so they
    // sit east of the positive ones, average the unwrapped values, then
    // wrap the result back into [-180, +180].
    let sum = 0
    for (const [x] of ring) sum += x < 0 ? x + 360 : x
    cx = sum / ring.length
    if (cx > 180) cx -= 360
  } else {
    cx = (xMin + xMax) / 2
  }
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
