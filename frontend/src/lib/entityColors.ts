export interface EntityColorEntry {
  hex: string;
  tailwind: string;
}

const ENTITY_COLORS: Record<string, EntityColorEntry> = {
  chemical:           { hex: '#2563EB', tailwind: 'blue-600' },
  species:            { hex: '#16A34A', tailwind: 'green-600' },
  plant_part:         { hex: '#84CC16', tailwind: 'lime-500' },
  extraction_method:  { hex: '#7C3AED', tailwind: 'violet-600' },
  development_stage:  { hex: '#F97316', tailwind: 'orange-500' },
  season:             { hex: '#EAB308', tailwind: 'yellow-500' },
  location:           { hex: '#06B6D4', tailwind: 'cyan-500' },
  bioactivity:        { hex: '#EC4899', tailwind: 'pink-500' },
  analytical_technique: { hex: '#64748B', tailwind: 'slate-500' },
  isolation_method:   { hex: '#8B5CF6', tailwind: 'violet-500' },
  disease:            { hex: '#DC2626', tailwind: 'red-600' },
  default:            { hex: '#6B7280', tailwind: 'gray-500' },
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
 * Display names for entity categories (used in legends, filters, headers).
 */
export const ENTITY_DISPLAY_NAMES: Record<string, string> = {
  chemical:               'Chemical',
  species:                'Species',
  plant_part:             'Plant part',
  extraction_method:      'Extraction method',
  development_stage:      'Development stage',
  season:                 'Season',
  location:               'Location',
  bioactivity:            'Bioactivity',
  analytical_technique:   'Analytical technique',
  isolation_method:       'Isolation method',
  disease:                'Disease',
};

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
  isolation_method:    { x: 0.50, y: 0.62 },
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
  isolation_method:    5,
  bioactivity:         6,
  disease:             7,
  location:            8,
  season:              9,
  development_stage:   10,
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
