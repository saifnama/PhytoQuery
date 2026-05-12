/**
 * Central entity color map.
 * Mirrors CSS --entity-*-rgb variables from index.css.
 * Used by KnowledgeGraph and ForceGraph3D.
 */

export interface EntityColorEntry {
  hex: string;
  rgb: string; // "r g b" (space-separated, matches CSS var format)
}

export const ENTITY_COLORS: Record<string, EntityColorEntry> = {
  default:            { hex: '#374151', rgb: '55 65 81' },
  chemical:           { hex: '#2563EB', rgb: '37 99 235' },
  species:            { hex: '#16A34A', rgb: '22 163 74' },
  plant_part:         { hex: '#84CC16', rgb: '132 204 22' },
  extraction_method:  { hex: '#7C3AED', rgb: '124 58 237' },
  development_stage:  { hex: '#F97316', rgb: '249 115 22' },
  season:             { hex: '#EAB308', rgb: '234 179 8' },
  location:           { hex: '#06B6D4', rgb: '6 182 212' },
  bioactivity:        { hex: '#EC4899', rgb: '236 72 153' },
  analytical_technique: { hex: '#64748B', rgb: '100 116 139' },
  disease:            { hex: '#DC2626', rgb: '220 38 38' },
};

/**
 * Returns the color for an entity label.
 * Falls back to 'default' if label is not in the map.
 */
export function getEntityColor(label: string): EntityColorEntry {
  const key = label.toLowerCase().replace(/\s+/g, '_');
  return ENTITY_COLORS[key] ?? ENTITY_COLORS.default;
}

/**
 * D3-force clustering target positions (0–1 normalized canvas space).
 * Semantic grouping: Source ← Process → Outcome + Context layer.
 */
export const ENTITY_CLUSTER_POSITIONS: Record<string, { x: number; y: number }> = {
  // Source layer
  chemical:           { x: 0.12, y: 0.50 },
  species:             { x: 0.22, y: 0.50 },
  // Intermediate
  plant_part:          { x: 0.40, y: 0.50 },
  // Process layer
  extraction_method:   { x: 0.58, y: 0.50 },
  analytical_technique: { x: 0.50, y: 0.38 },
  // Outcome layer
  bioactivity:         { x: 0.78, y: 0.38 },
  disease:             { x: 0.82, y: 0.62 },
  // Context layer
  location:            { x: 0.35, y: 0.22 },
  season:              { x: 0.35, y: 0.78 },
  development_stage:   { x: 0.22, y: 0.72 },
};

export const ENTITY_CATEGORY_INDEX: Record<string, number> = {
  chemical:            0,
  species:             1,
  plant_part:          2,
  extraction_method:   3,
  analytical_technique: 4,
  bioactivity:         5,
  disease:             6,
  location:            7,
  season:              8,
  development_stage:   9,
};

/**
 * Get ECharts series itemStyle for a given entity label.
 */
export function getEchartsItemStyle(label: string) {
  const color = getEntityColor(label);
  return {
    color: color.hex,
  };
}