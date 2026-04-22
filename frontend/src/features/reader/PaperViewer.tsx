import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { sanitizeHtml } from '../../utils/sanitize';
import { DownloadSimple, PencilSimple } from '@phosphor-icons/react';
import type { Entity, TocItem } from '../../types';
import SmilesDrawer from 'smiles-drawer';

const SPECIES_SELECTOR = '.ent-species, mark.ner-species';
const CHEMICAL_SELECTOR = '.ent-chemical, mark.ner-chemical';
const SPECIES_POPUP_WIDTH = 320;
const SPECIES_POPUP_ESTIMATED_HEIGHT = 280;

const ENTITY_GROUP_ORDER = [
  'CHEMICAL',
  'SPECIES',
  'PLANT PART',
  'DEVELOPMENT STAGE',
  'EXTRACTION METHOD',
  'ANALYTICAL TECHNIQUE',
  'BIOACTIVITY',
  'DISEASE',
  'SEASON',
  'LOCATION',
] as const;

type EntityGroupLabel = (typeof ENTITY_GROUP_ORDER)[number];

const ENTITY_GROUP_CONFIG: Record<EntityGroupLabel, {
  accentVar: string;
  highlightSelector: string;
}> = {
  'CHEMICAL': {
    accentVar: '--entity-chemical',
    highlightSelector: '.ent-chemical, mark.ner-chemical',
  },
  'SPECIES': {
    accentVar: '--entity-species',
    highlightSelector: '.ent-species, mark.ner-species',
  },
  'PLANT PART': {
    accentVar: '--entity-plant-part',
    highlightSelector: '.ent-plant-part, mark.ner-plant-part',
  },
  'DEVELOPMENT STAGE': {
    accentVar: '--entity-development-stage',
    highlightSelector: '.ent-development-stage, mark.ner-development-stage',
  },
  'EXTRACTION METHOD': {
    accentVar: '--entity-extraction-method',
    highlightSelector: '.ent-extraction-method, mark.ner-extraction-method',
  },
  'ANALYTICAL TECHNIQUE': {
    accentVar: '--entity-analytical-technique',
    highlightSelector: '.ent-analytical-technique, mark.ner-analytical-technique',
  },
  'BIOACTIVITY': {
    accentVar: '--entity-bioactivity',
    highlightSelector: '.ent-bioactivity, mark.ner-bioactivity',
  },
  'DISEASE': {
    accentVar: '--entity-disease',
    highlightSelector: '.ent-disease, mark.ner-disease',
  },
  'SEASON': {
    accentVar: '--entity-season',
    highlightSelector: '.ent-season, mark.ner-season',
  },
  'LOCATION': {
    accentVar: '--entity-location',
    highlightSelector: '.ent-location, mark.ner-location',
  },
};

const createInitialExpandedGroups = (grouped: GroupedEntities) => {
  return ENTITY_GROUP_ORDER.reduce<Record<EntityGroupLabel, boolean>>((acc, label) => {
    acc[label] = (grouped[label]?.length ?? 0) > 0;
    return acc;
  }, {} as Record<EntityGroupLabel, boolean>);
};

const createInitialEnabledHighlightGroups = () => {
  return ENTITY_GROUP_ORDER.reduce<Record<EntityGroupLabel, boolean>>((acc, label) => {
    acc[label] = true;
    return acc;
  }, {} as Record<EntityGroupLabel, boolean>);
};

const getEntityGroupPanelId = (label: EntityGroupLabel) => `entity-group-panel-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;

type SpeciesPopupData = {
  primaryName: string;
  acceptedScientificName?: string;
  scientificNameVerified?: string;
  commonName?: string;
  canonical?: string;
  sourceDb?: string;
  sourceUrl?: string;
  taxonId?: string;
  matchStatus?: string;
  reviewRequired?: string;
  nameType?: Entity['name_type'];
  metadataScore: number;
};

type SpeciesPopupState = {
  species: SpeciesPopupData;
  anchorText: string;
  position: {
    top: number;
    left: number;
  };
};

type ChemicalPopupData = {
  primaryName: string;
  preferredName?: string;
  synonyms?: string[];
  smiles?: string;
  inchikey?: string;
  molecularFormula?: string;
  sourceDb?: string;
  sourceUrl?: string;
};

type ChemicalPopupState = {
  chemical: ChemicalPopupData;
  anchorText: string;
  position: {
    top: number;
    left: number;
  };
};

const stripHtml = (value?: string | null) => (value || '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

const normalizeLookupText = (value?: string | null) => stripHtml(value).toLowerCase();

const getSpeciesPrimaryName = (entity: Entity) => {
  const acceptedScientificName = stripHtml(entity.accepted_scientific_name);
  const verifiedScientificName = stripHtml(entity.scientific_name_verified);
  const canonicalName = stripHtml(entity.canonical);
  const commonName = stripHtml(entity.common_name);
  const cleanText = stripHtml(entity.text);

  if (acceptedScientificName) return acceptedScientificName;
  if (verifiedScientificName) return verifiedScientificName;
  if (entity.name_type === 'scientific' && canonicalName) return canonicalName;
  if (canonicalName && normalizeLookupText(canonicalName) !== normalizeLookupText(commonName)) {
    return canonicalName;
  }
  return canonicalName || cleanText;
};

const getSpeciesMetadataScore = (entity: Entity) => {
  let score = 0;
  if (stripHtml(entity.accepted_scientific_name)) score += 32;
  if (stripHtml(entity.scientific_name_verified)) score += 16;
  if (entity.name_type === 'scientific') score += 8;
  if (stripHtml(entity.common_name)) score += 4;
  if (stripHtml(entity.canonical)) score += 2;
  if (stripHtml(entity.source_db)) score += 2;
  if (stripHtml(entity.source_url)) score += 2;
  if (stripHtml(entity.taxon_id)) score += 2;
  if (stripHtml(entity.match_status)) score += 1;
  if (stripHtml(entity.review_required)) score += 1;
  return score;
};

const pickBetterSpeciesRepresentative = (current: Entity, candidate: Entity) => {
  const currentScore = getSpeciesMetadataScore(current);
  const candidateScore = getSpeciesMetadataScore(candidate);

  if (candidateScore > currentScore) return candidate;
  if (candidateScore < currentScore) return current;

  return stripHtml(candidate.text).length > stripHtml(current.text).length ? candidate : current;
};

const getSpeciesAliasList = (entity: Entity) => {
  const aliasSet = new Set<string>();
  const addAlias = (value?: string | null) => {
    const cleaned = stripHtml(value);
    if (cleaned) aliasSet.add(cleaned);
  };

  addAlias(entity.text);
  addAlias(entity.accepted_scientific_name);
  addAlias(entity.scientific_name_verified);
  addAlias(entity.canonical);
  addAlias(entity.common_name);
  (entity.aliases || []).forEach((alias) => addAlias(alias));

  return Array.from(aliasSet);
};

const buildSpeciesPopupData = (entity: Entity): SpeciesPopupData => {
  return {
    primaryName: getSpeciesPrimaryName(entity),
    acceptedScientificName: stripHtml(entity.accepted_scientific_name) || undefined,
    scientificNameVerified: stripHtml(entity.scientific_name_verified) || undefined,
    commonName: stripHtml(entity.common_name) || undefined,
    canonical: stripHtml(entity.canonical) || undefined,
    sourceDb: stripHtml(entity.source_db) || undefined,
    sourceUrl: stripHtml(entity.source_url) || undefined,
    taxonId: stripHtml(entity.taxon_id) || undefined,
    matchStatus: stripHtml(entity.match_status) || undefined,
    reviewRequired: stripHtml(entity.review_required) || undefined,
    nameType: entity.name_type ?? undefined,
    metadataScore: getSpeciesMetadataScore(entity),
  };
};

const getSpeciesPopupPosition = (anchor: HTMLElement) => {
  const rect = anchor.getBoundingClientRect();
  const margin = 12;
  const top = rect.bottom + SPECIES_POPUP_ESTIMATED_HEIGHT + margin <= window.innerHeight
    ? rect.bottom + 10
    : Math.max(margin, rect.top - SPECIES_POPUP_ESTIMATED_HEIGHT - 10);
  const centeredLeft = rect.left + rect.width / 2 - SPECIES_POPUP_WIDTH / 2;
  const left = Math.min(
    Math.max(margin, centeredLeft),
    Math.max(margin, window.innerWidth - SPECIES_POPUP_WIDTH - margin)
  );

  return { top, left };
};


interface PaperViewerProps {
  doi: string;
  mode: 'full_text' | 'abstract';
  title: string;
  html: string;
  toc: TocItem[];
  entities?: Entity[];
  isExtracted?: boolean;
  isExtracting?: boolean;
  extractionError?: string | null;
  fallbackSource?: { source: string; url: string };
  isFetchingFallback?: boolean;
  paperAuthors?: string[];
  paperJournal?: string;
  paperDate?: string;
  onExtract?: () => void;
}

interface GroupedEntities {
  [label: string]: {
    text: string;
    count: number;
    aliases: string[];
    subtitle?: string;
    // Chemical metadata
    preferred_name?: string;
    inchikey?: string;
    smiles?: string;
    molecular_formula?: string;
    source_db?: string;
    source_url?: string;
  }[];
}

const isChemicalLikeLabel = (label: string) => label === 'CHEMICAL';

const PaperViewer: React.FC<PaperViewerProps> = ({
   doi,
   mode,
   title,
   html,
   toc,
   entities = [],
   isExtracted = false,
   isExtracting = false,
   extractionError = null,
   fallbackSource,
   isFetchingFallback = false,
   paperAuthors = [],
    paperJournal,
    paperDate,
    onExtract,
   }) => {
  const [activeHeading, setActiveHeading] = useState<string | null>(null);
  const [currentSectionIdx, setCurrentSectionIdx] = useState(0);
  const [entitySearch, setEntitySearch] = useState('');
  const [expandedChemical, setExpandedChemical] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Record<EntityGroupLabel, boolean>>(() => createInitialExpandedGroups({}));
  const [enabledHighlightGroups, setEnabledHighlightGroups] = useState<Record<EntityGroupLabel, boolean>>(() => createInitialEnabledHighlightGroups());
  const [activeSpeciesPopup, setActiveSpeciesPopup] = useState<SpeciesPopupState | null>(null);
  const [activeChemicalPopup, setActiveChemicalPopup] = useState<ChemicalPopupState | null>(null);
  const titleContainerRef = useRef<HTMLHeadingElement>(null);
  const htmlContainerRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);
  const speciesPopupRef = useRef<HTMLDivElement>(null);
  const chemicalPopupRef = useRef<HTMLDivElement>(null);
  const chemicalStructureSvgRef = useRef<SVGSVGElement | null>(null);
  const chemicalStructureRenderIdRef = useRef(0);
  const activeSpeciesAnchorRef = useRef<HTMLElement | null>(null);
  const activeChemicalAnchorRef = useRef<HTMLElement | null>(null);
  const chemicalPopupRequestIdRef = useRef(0);
  const [isRenderingChemicalStructure, setIsRenderingChemicalStructure] = useState(false);
  const [chemicalStructureError, setChemicalStructureError] = useState(false);

  const getInteractiveRoots = useCallback(() => {
    const roots: HTMLElement[] = [];
    if (titleContainerRef.current) {
      roots.push(titleContainerRef.current);
    }
    if (htmlContainerRef.current) {
      roots.push(htmlContainerRef.current);
    }
    return roots;
  }, []);

  const closeSpeciesPopup = useCallback(() => {
    activeSpeciesAnchorRef.current = null;
    setActiveSpeciesPopup(null);
  }, []);

  const closeChemicalPopup = useCallback(() => {
    chemicalPopupRequestIdRef.current += 1;
    activeChemicalAnchorRef.current = null;
    setIsRenderingChemicalStructure(false);
    setChemicalStructureError(false);
    setExpandedChemical(null);
    setActiveChemicalPopup(null);
  }, []);

  const toggleExpandedChemical = useCallback((key: string) => {
    setExpandedChemical(prev => prev === key ? null : key);
  }, []);

  const toggleEntityGroup = useCallback((label: EntityGroupLabel) => {
    setExpandedGroups((current) => ({
      ...current,
      [label]: !current[label],
    }));
  }, []);

  const toggleHighlightGroup = useCallback((label: EntityGroupLabel) => {
    setEnabledHighlightGroups((current) => ({
      ...current,
      [label]: !current[label],
    }));
  }, []);

  const isExpandedChemical = activeChemicalPopup && expandedChemical === activeChemicalPopup.chemical.primaryName;

  const speciesLookup = useMemo(() => {
    const groupedSpecies = new Map<string, { representative: Entity; aliases: Set<string> }>();

    entities.forEach((entity) => {
      if (entity.label !== 'SPECIES') return;

      const groupKey = normalizeLookupText(getSpeciesPrimaryName(entity))
        || normalizeLookupText(entity.canonical)
        || normalizeLookupText(entity.text);

      if (!groupKey) return;

      const existingGroup = groupedSpecies.get(groupKey);
      if (!existingGroup) {
        groupedSpecies.set(groupKey, {
          representative: entity,
          aliases: new Set(getSpeciesAliasList(entity)),
        });
        return;
      }

      existingGroup.representative = pickBetterSpeciesRepresentative(existingGroup.representative, entity);
      // Merge ALL aliases from both entities (not just new ones)
      const allAliases = new Set([
        ...getSpeciesAliasList(existingGroup.representative),
        ...getSpeciesAliasList(entity)
      ]);
      existingGroup.aliases = allAliases;
    });

    const lookup = new Map<string, SpeciesPopupData>();

    groupedSpecies.forEach(({ representative, aliases }) => {
      const speciesData = buildSpeciesPopupData(representative);

      // Add ALL aliases to lookup so popup works regardless of which variant is clicked
      aliases.forEach((alias) => {
        const normalizedAlias = normalizeLookupText(alias);
        if (!normalizedAlias) return;

        const existing = lookup.get(normalizedAlias);
        if (!existing || speciesData.metadataScore > existing.metadataScore) {
          lookup.set(normalizedAlias, speciesData);
        }
      });
      
      // Also add the primary name itself
      const primaryName = normalizeLookupText(speciesData.primaryName);
      if (primaryName) {
        const existing = lookup.get(primaryName);
        if (!existing || speciesData.metadataScore > existing.metadataScore) {
          lookup.set(primaryName, speciesData);
        }
      }
    });

    return lookup;
  }, [entities]);

  const openSpeciesPopup = useCallback((target: HTMLElement) => {
    const text = stripHtml(target.textContent);
    if (!text) return;
    
    // Try to find in lookup - check data attribute first, then text variations
    let species = speciesLookup.get(target.dataset.speciesLookupKey || "");
    if (!species) {
      species = speciesLookup.get(text.toLowerCase());
    }
    if (!species) {
      species = speciesLookup.get(normalizeLookupText(text));
    }
    if (!species) {
      // Try each word in text (for multi-word species names)
      for (const [key, val] of speciesLookup) {
        if (text.toLowerCase().includes(key) || key.includes(text.toLowerCase())) {
          species = val;
          break;
        }
      }
    }
    if (!species) return;

    activeSpeciesAnchorRef.current = target;
    setActiveSpeciesPopup({
      species,
      anchorText: text,
      position: getSpeciesPopupPosition(target),
    });
  }, [speciesLookup]);

  // Chemical lookup map - handles chemical entities for popup/sidebar metadata
  const chemicalLookup = useMemo(() => {
    const groupedChemicals = new Map<string, { representative: Entity; aliases: Set<string> }>();
    
    entities.forEach((entity) => {
      if (entity.label !== 'CHEMICAL') return;
      
      const groupKey = normalizeLookupText(entity.canonical || entity.text);
      if (!groupKey) return;
      
      const existingGroup = groupedChemicals.get(groupKey);
      if (!existingGroup) {
        groupedChemicals.set(groupKey, {
          representative: entity,
          aliases: new Set((entity.aliases || []).map(a => stripHtml(a))),
        });
        return;
      }
      
      // Keep the one with more metadata
      const existingMetaCount = [existingGroup.representative.inchikey, existingGroup.representative.molecular_formula, existingGroup.representative.source_db, existingGroup.representative.smiles].filter(Boolean).length;
      const newMetaCount = [entity.inchikey, entity.molecular_formula, entity.source_db, entity.smiles].filter(Boolean).length;
      if (newMetaCount > existingMetaCount) {
        groupedChemicals.set(groupKey, { representative: entity, aliases: existingGroup.aliases });
      } else {
        // Merge aliases even if not replacing
        (entity.aliases || []).forEach((alias) => existingGroup.aliases.add(stripHtml(alias)));
      }
    });
    
    const lookup = new Map<string, ChemicalPopupData>();
    
    groupedChemicals.forEach(({ representative, aliases }) => {
      // Use canonical if available, otherwise fall back to text
      const keyText = representative.canonical || representative.text;
      const normalizedText = normalizeLookupText(keyText);
      if (!normalizedText) return;
      
      // Create popup data
      const chemicalData: ChemicalPopupData = {
        primaryName: keyText,  // Use the key text (canonical or text)
        preferredName: stripHtml(representative.preferred_name) || undefined,
        synonyms: Array.from(aliases).filter(a => normalizeLookupText(a) !== normalizedText),
        smiles: stripHtml(representative.smiles) || undefined,
        inchikey: stripHtml(representative.inchikey) || undefined,
        molecularFormula: stripHtml(representative.molecular_formula) || undefined,
        sourceDb: stripHtml(representative.source_db) || undefined,
        sourceUrl: stripHtml(representative.source_url) || undefined,
      };
      
      // Add the canonical/primary text
      lookup.set(normalizedText, chemicalData);
      
      // Add all aliases so popup works when clicking any variant
      aliases.forEach((alias) => {
        const normalizedAlias = normalizeLookupText(alias);
        if (normalizedAlias && normalizedAlias !== normalizedText) {
          lookup.set(normalizedAlias, chemicalData);
        }
      });
    });
    
    return lookup;
  }, [entities]);

  const openChemicalPopup = useCallback((target: HTMLElement) => {
    const requestId = chemicalPopupRequestIdRef.current + 1;
    chemicalPopupRequestIdRef.current = requestId;
    activeChemicalAnchorRef.current = target;

    // Try to find in lookup - check data attribute first, then text variations
    const text = stripHtml(target.textContent);
    if (!text) return;
    
    let chemical = chemicalLookup.get(target.dataset.chemicalLookupKey || "");
    if (!chemical) {
      chemical = chemicalLookup.get(text.toLowerCase());
    }
    if (!chemical) {
      chemical = chemicalLookup.get(normalizeLookupText(text));
    }
    if (!chemical) {
      // Try each word in text
      for (const [key, val] of chemicalLookup) {
        if (text.toLowerCase().includes(key) || key.includes(text.toLowerCase())) {
          chemical = val;
          break;
        }
      }
    }
    if (!chemical) return;
    if (requestId !== chemicalPopupRequestIdRef.current || !target.isConnected) {
      return;
    }

    setIsRenderingChemicalStructure(!!chemical.smiles);
    setChemicalStructureError(false);
    setActiveChemicalPopup({
      chemical,
      anchorText: stripHtml(target.textContent),
      position: getSpeciesPopupPosition(target),
    });
  }, [chemicalLookup]);

  useEffect(() => {
    const svgElement = chemicalStructureSvgRef.current;
    const smiles = activeChemicalPopup?.chemical.smiles;
    const renderId = chemicalStructureRenderIdRef.current + 1;
    chemicalStructureRenderIdRef.current = renderId;

    if (svgElement) {
      while (svgElement.firstChild) {
        svgElement.removeChild(svgElement.firstChild);
      }
    }

    if (!activeChemicalPopup || !smiles || !svgElement) {
      setIsRenderingChemicalStructure(false);
      setChemicalStructureError(false);
      return;
    }

    let cancelled = false;
    setIsRenderingChemicalStructure(true);
    setChemicalStructureError(false);

    const drawer = new SmilesDrawer.SvgDrawer({ width: 400, height: 300 });

    SmilesDrawer.parse(
      smiles,
      (tree: unknown) => {
        if (cancelled || chemicalStructureRenderIdRef.current !== renderId || !chemicalStructureSvgRef.current) {
          return;
        }

        const currentSvg = chemicalStructureSvgRef.current;
        while (currentSvg.firstChild) {
          currentSvg.removeChild(currentSvg.firstChild);
        }

        try {
          drawer.draw(tree, currentSvg, 'light');
          const hasVisibleStructure = currentSvg.querySelector('path, text, line, circle, polygon, rect') !== null;
          if (cancelled || chemicalStructureRenderIdRef.current !== renderId) {
            return;
          }
          setChemicalStructureError(!hasVisibleStructure);
          setIsRenderingChemicalStructure(false);
        } catch (error) {
          console.error('Failed to draw molecule:', error, 'SMILES:', smiles);
          if (cancelled || chemicalStructureRenderIdRef.current !== renderId) {
            return;
          }
          setIsRenderingChemicalStructure(false);
          setChemicalStructureError(true);
        }
      },
      (error: unknown) => {
        if (cancelled || chemicalStructureRenderIdRef.current !== renderId) {
          return;
        }
        console.error('Failed to parse molecule:', error, 'SMILES:', smiles);
        setIsRenderingChemicalStructure(false);
        setChemicalStructureError(true);
      },
    );

    return () => {
      cancelled = true;
      if (chemicalStructureRenderIdRef.current === renderId) {
        chemicalStructureRenderIdRef.current += 1;
      }
    };
  }, [activeChemicalPopup]);

  // Group entities by label and calculate true frequency from frontend HTML
  const groupedEntities = useCallback(() => {
    const grouped: GroupedEntities = {};
    if (!entities || entities.length === 0) return grouped;

    // Create a plain text version of the HTML to count occurrences accurately without HTML tag interference
    let fullText = '';
    if (html) {
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = html;
      fullText = tempDiv.textContent || tempDiv.innerText || '';
    }

    // Group entities by normalized key but track all original texts
    const uniqueEntities: Record<string, {
      label: string;
      originalTexts: Record<string, number>;
      aliases: Set<string>;
      representative: Entity;
    }> = {};

    entities.forEach((entity) => {
      // Use canonical form from backend, or fallback to normalized text
      const cleanText = stripHtml(entity.text);
      const canonicalForm = entity.label === 'SPECIES'
        ? getSpeciesPrimaryName(entity)
        : (stripHtml(entity.canonical) || cleanText);
      const normalizedKey = canonicalForm.toLowerCase();
      const groupKey = `${entity.label}::${normalizedKey}`;
      if (!uniqueEntities[groupKey]) {
        uniqueEntities[groupKey] = { 
          label: entity.label, 
          originalTexts: {},
          aliases: new Set<string>(),
          representative: entity,
        };
      } else if (entity.label === 'SPECIES') {
        uniqueEntities[groupKey].representative = pickBetterSpeciesRepresentative(
          uniqueEntities[groupKey].representative,
          entity
        );
      }

      if (entity.label === 'SPECIES') {
        getSpeciesAliasList(entity).forEach((alias) => uniqueEntities[groupKey].aliases.add(alias));
      } else {
        (entity.aliases || []).forEach((alias) => uniqueEntities[groupKey].aliases.add(alias));
        uniqueEntities[groupKey].aliases.add(canonicalForm);
      }

      // Store original text for display and frequency tracking
      uniqueEntities[groupKey].originalTexts[cleanText] = (uniqueEntities[groupKey].originalTexts[cleanText] || 0) + 1;
    });

    Object.keys(uniqueEntities).forEach((groupKey) => {
      const data = uniqueEntities[groupKey];
      const representative = data.representative;
      
      // Count frequency of EACH variant in full text to find most frequent form
      const aliasList = Array.from(data.aliases);
      const variantCounts: Record<string, number> = {};
      
      if (fullText) {
        for (const variant of aliasList) {
          const normalizedVariant = variant.toLowerCase();
          let count = 0;
          try {
            const escapedText = normalizedVariant.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`(?:^|\\W)(${escapedText})(?:$|\\W)`, 'gi');
            while (regex.exec(fullText) !== null) {
              count++;
              if (regex.lastIndex > 0) {
                regex.lastIndex -= 1;
              }
            }
          } catch {
            // Skip invalid regex
          }
          if (count > 0) {
            variantCounts[variant] = count;
          }
        }
      }
      
      // Find the most frequent variant in the full text for display
      const mostFrequentText = Object.entries(variantCounts)
        .sort(([, a], [, b]) => b - a)[0]?.[0] || stripHtml(representative.canonical) || stripHtml(representative.text);
      
      // Calculate total count from all variants
      let trueCount = Object.values(variantCounts).reduce((sum, c) => sum + c, 0);

      // Fallback to at least 1 if it was extracted by AI
      if (trueCount === 0) {
        trueCount = 1;
      }

      if (!grouped[data.label]) {
        grouped[data.label] = [];
      }
      const displayText = data.label === 'SPECIES'
        ? getSpeciesPrimaryName(representative)
        : mostFrequentText;
      // No subtitle for species - show only scientific name
      const subtitle = undefined;
      // Use most frequent text form for display
      const chemicalMeta = isChemicalLikeLabel(data.label) ? {
        preferred_name: representative.preferred_name,
        inchikey: representative.inchikey,
        smiles: representative.smiles,
        molecular_formula: representative.molecular_formula,
        source_db: representative.source_db,
        source_url: representative.source_url,
      } : {};
      grouped[data.label].push({ text: displayText, count: trueCount, aliases: aliasList, subtitle, ...chemicalMeta });
    });

    // Sort by count descending
    Object.keys(grouped).forEach(label => {
      grouped[label].sort((a, b) => b.count - a.count);
    });

    return grouped;
  }, [entities, html]);

  const groupedEntityMap = useMemo(() => groupedEntities(), [groupedEntities]);
  const normalizedEntitySearch = entitySearch.trim().toLowerCase();

  const visibleGroupedEntities = useMemo(() => {
    return ENTITY_GROUP_ORDER.map((label) => {
      const items = groupedEntityMap[label] ?? [];
      const visibleItems = !normalizedEntitySearch
        ? items
        : items.filter((ent) => {
            const displayMatch = ent.text.toLowerCase().includes(normalizedEntitySearch);
            const aliasMatch = ent.aliases.some((alias) => {
              const normalizedAlias = alias.toLowerCase();
              return normalizedAlias !== ent.text.toLowerCase() && normalizedAlias.includes(normalizedEntitySearch);
            });

            return displayMatch || aliasMatch;
          });

      return {
        label,
        items,
        visibleItems,
        totalCount: items.reduce((sum, item) => sum + item.count, 0),
        termCount: items.length,
        ...ENTITY_GROUP_CONFIG[label],
      };
    });
  }, [groupedEntityMap, normalizedEntitySearch]);

  const hasVisibleEntityMatches = useMemo(() => {
    return visibleGroupedEntities.some((group) => group.visibleItems.length > 0);
  }, [visibleGroupedEntities]);

  const disabledHighlightGroups = useMemo(() => {
    return ENTITY_GROUP_ORDER.filter((label) => !enabledHighlightGroups[label]);
  }, [enabledHighlightGroups]);

  const disabledHighlightGroupData = disabledHighlightGroups.join(' ') || undefined;
  
  
  
  const resetReaderUiState = useCallback(() => {
    setEntitySearch('');
    setExpandedChemical(null);
    setExpandedGroups(createInitialExpandedGroups(isExtracted ? groupedEntityMap : {}));
    setEnabledHighlightGroups(createInitialEnabledHighlightGroups());
    closeSpeciesPopup();
    closeChemicalPopup();
  }, [isExtracted, groupedEntityMap, closeSpeciesPopup, closeChemicalPopup]);

  useEffect(() => {
    const resetTimer = window.setTimeout(() => {
      resetReaderUiState();
    }, 0);

    return () => window.clearTimeout(resetTimer);
  }, [doi, html, resetReaderUiState]);

  useEffect(() => {
    const roots = getInteractiveRoots();
    if (roots.length === 0) return;

    roots.forEach((container) => {
      const speciesNodes = container.matches(SPECIES_SELECTOR)
        ? [container as HTMLElement]
        : Array.from(container.querySelectorAll<HTMLElement>(SPECIES_SELECTOR));

      speciesNodes.forEach((node) => {
        const lookupKey = normalizeLookupText(node.textContent);
        const species = speciesLookup.get(lookupKey);

        if (!species) {
          node.removeAttribute('data-species-lookup-key');
          node.removeAttribute('role');
          node.removeAttribute('tabindex');
          node.removeAttribute('aria-haspopup');
          node.removeAttribute('aria-expanded');
          return;
        }

        node.dataset.speciesLookupKey = lookupKey;
        node.setAttribute('role', 'button');
        node.setAttribute('tabindex', '0');
        node.setAttribute('aria-haspopup', 'dialog');
        node.setAttribute('aria-expanded', activeSpeciesAnchorRef.current === node && activeSpeciesPopup ? 'true' : 'false');
        node.setAttribute('title', species.primaryName);
      });
    });
  }, [getInteractiveRoots, html, speciesLookup, activeSpeciesPopup]);

  useEffect(() => {
    const roots = getInteractiveRoots();
    if (roots.length === 0) return;

    const handleSpeciesClick = (event: MouseEvent) => {
      const clickedEl = event.target as HTMLElement;
      
      // Only handle clicks on highlighted species elements
      if (!clickedEl.classList.contains('ent-species') && !clickedEl.closest('.ent-species')) {
        return;
      }
      
      // Get the text content and lookup in speciesLookup
      const textContent = clickedEl.textContent?.trim();
      if (!textContent) return;
      
      const normalizedKey = normalizeLookupText(textContent);
      const species = speciesLookup.get(normalizedKey);
      
      if (!species) {
        const species2 = speciesLookup.get(textContent.toLowerCase());
        if (!species2) return;
      }
      
      event.preventDefault();

      // Use clicked element directly
      const targetEl = clickedEl.closest('.ent-species') || clickedEl;

      if (activeSpeciesAnchorRef.current === targetEl && activeSpeciesPopup) {
        closeSpeciesPopup();
        return;
      }

      openSpeciesPopup(targetEl as HTMLElement);
    };

    const handleSpeciesKeyDown = (event: KeyboardEvent) => {
      const target = (event.target as HTMLElement).closest('.ent-species') as HTMLElement | null;
      if (!target || !roots.some((root) => root.contains(target))) return;

      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openSpeciesPopup(target);
      }

      if (event.key === 'Escape') {
        closeSpeciesPopup();
        target.blur();
      }
    };

    roots.forEach((container) => {
      container.addEventListener('click', handleSpeciesClick);
      container.addEventListener('keydown', handleSpeciesKeyDown);
    });

    return () => {
      roots.forEach((container) => {
        container.removeEventListener('click', handleSpeciesClick);
        container.removeEventListener('keydown', handleSpeciesKeyDown);
      });
    };
  }, [activeSpeciesPopup, closeSpeciesPopup, getInteractiveRoots, openSpeciesPopup, speciesLookup]);

  useEffect(() => {
    if (!activeSpeciesPopup) return;

    const repositionPopup = () => {
      const anchor = activeSpeciesAnchorRef.current;
      if (!anchor) return;

      const nextPosition = getSpeciesPopupPosition(anchor);
      setActiveSpeciesPopup((current) => {
        if (!current) return current;
        if (current.position.top === nextPosition.top && current.position.left === nextPosition.left) {
          return current;
        }
        return { ...current, position: nextPosition };
      });
    };

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (speciesPopupRef.current?.contains(target)) return;
      if (activeSpeciesAnchorRef.current?.contains(target)) return;
      closeSpeciesPopup();
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeSpeciesPopup();
      }
    };

    window.addEventListener('resize', repositionPopup);
    window.addEventListener('scroll', repositionPopup, true);
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleEscape);

    return () => {
      window.removeEventListener('resize', repositionPopup);
      window.removeEventListener('scroll', repositionPopup, true);
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [activeSpeciesPopup, closeSpeciesPopup]);

  // Chemical popup - add data attributes to highlighted chemicals
  useEffect(() => {
    const roots = getInteractiveRoots();
    if (roots.length === 0) return;

    roots.forEach((container) => {
      const chemicalNodes = container.matches(CHEMICAL_SELECTOR)
        ? [container as HTMLElement]
        : Array.from(container.querySelectorAll<HTMLElement>(CHEMICAL_SELECTOR));

      chemicalNodes.forEach((node) => {
        const lookupKey = normalizeLookupText(node.textContent);
        const chemical = chemicalLookup.get(lookupKey);

        const fallbackKey = normalizeLookupText(node.textContent || '');
        const chemicalFallback = !chemical ? chemicalLookup.get(fallbackKey) : chemical;

        if (!chemicalFallback) {
          node.removeAttribute('data-chemical-lookup-key');
          node.removeAttribute('role');
          node.removeAttribute('tabindex');
          node.removeAttribute('aria-haspopup');
          node.removeAttribute('aria-expanded');
          return;
        }

        node.dataset.chemicalLookupKey = lookupKey;
        node.setAttribute('role', 'button');
        node.setAttribute('tabindex', '0');
        node.setAttribute('aria-haspopup', 'dialog');
        node.setAttribute('aria-expanded', activeChemicalAnchorRef.current === node && activeChemicalPopup ? 'true' : 'false');
        node.setAttribute('title', chemicalFallback.primaryName);
      });
    });
  }, [getInteractiveRoots, html, chemicalLookup, activeChemicalPopup]);

  // Chemical popup click/key handlers
  useEffect(() => {
    const roots = getInteractiveRoots();
    if (roots.length === 0) return;

    const handleChemicalClick = (event: MouseEvent) => {
      const clickedEl = event.target as HTMLElement;
      
      // Only handle clicks on highlighted chemical elements
      if (!clickedEl.classList.contains('ent-chemical') && !clickedEl.closest('.ent-chemical')) {
        return;
      }
      
      // Get the text content and lookup in chemicalLookup
      const textContent = clickedEl.textContent?.trim();
      if (!textContent) return;
      
      const normalizedKey = normalizeLookupText(textContent);
      const chemical = chemicalLookup.get(normalizedKey);
      
      if (!chemical) {
        // Try with original text as fallback
        const chemical2 = chemicalLookup.get(textContent.toLowerCase());
        if (!chemical2) return;
      }
      
      event.preventDefault();

      // Use clicked element directly - wrap to provide needed interface
      const targetEl = clickedEl.closest('.ent-chemical') || clickedEl;
      
      if (activeChemicalAnchorRef.current === targetEl && activeChemicalPopup) {
        closeChemicalPopup();
        return;
      }

      openChemicalPopup(targetEl as HTMLElement);
    };

    const handleChemicalKeyDown = (event: KeyboardEvent) => {
      const target = (event.target as HTMLElement).closest<HTMLElement>('[data-chemical-lookup-key], .ent-chemical');
      if (!target || !roots.some((root) => root.contains(target))) return;

      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openChemicalPopup(target);
      }

      if (event.key === 'Escape') {
        closeChemicalPopup();
        target.blur();
      }
    };

    roots.forEach((container) => {
      container.addEventListener('click', handleChemicalClick);
      container.addEventListener('keydown', handleChemicalKeyDown);
    });

    return () => {
      roots.forEach((container) => {
        container.removeEventListener('click', handleChemicalClick);
        container.removeEventListener('keydown', handleChemicalKeyDown);
      });
    };
  }, [activeChemicalPopup, closeChemicalPopup, getInteractiveRoots, openChemicalPopup, chemicalLookup]);

  // Chemical popup repositioning
  useEffect(() => {
    const repositionPopup = () => {
      const anchor = activeChemicalAnchorRef.current;
      if (!anchor || !activeChemicalPopup) return;

      const nextPosition = getSpeciesPopupPosition(anchor);
      setActiveChemicalPopup((current) => {
        if (!current) return current;
        return { ...current, position: nextPosition };
      });
    };

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      const activeAnchor = activeChemicalAnchorRef.current;
      if (!activeAnchor) return;
      if (chemicalPopupRef.current?.contains(target)) return;
      if (activeAnchor.contains(target)) return;
      closeChemicalPopup();
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeChemicalPopup();
      }
    };

    window.addEventListener('resize', repositionPopup);
    window.addEventListener('scroll', repositionPopup, true);
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleEscape);

    return () => {
      window.removeEventListener('resize', repositionPopup);
      window.removeEventListener('scroll', repositionPopup, true);
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [activeChemicalPopup, closeChemicalPopup]);

  // Setup scroll spy for sub-headings (based on HTML blob sections)
  useEffect(() => {
    if (!htmlContainerRef.current) return;
    const container = htmlContainerRef.current;
    const sectionNodes = container.querySelectorAll('section');
    if (sectionNodes.length === 0) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.intersectionRatio > 0) {
            const id = entry.target.id;
            const match = id.match(/^section-(\d+)$/);
            if (match) {
              setCurrentSectionIdx(parseInt(match[1], 10));
            } else {
              const idx = Array.from(sectionNodes).indexOf(entry.target as HTMLElement);
              if (idx >= 0) setCurrentSectionIdx(idx);
            }
            setActiveHeading(id);
          }
        });
      },
      { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
    );

    sectionNodes.forEach((el: Element) => observerRef.current?.observe(el as Element));

    return () => {
      observerRef.current?.disconnect();
    };
  }, [html]);

  // Scroll to element by ID
  const scrollToId = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Keyboard shortcuts (navigate through TOC items)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((toc?.length ?? 0) === 0) return;
      if (e.key === 'ArrowRight') {
        if (currentSectionIdx < (toc?.length ?? 0) - 1) {
          const nextId = toc?.[currentSectionIdx + 1]?.id;
          if (nextId) scrollToId(nextId);
        }
      } else if (e.key === 'ArrowLeft') {
        if (currentSectionIdx > 0) {
          const prevId = toc?.[currentSectionIdx - 1]?.id;
          if (prevId) scrollToId(prevId);
        }
      } else if (e.key === 'ArrowUp') {
        window.scrollBy({ top: -100, behavior: 'smooth' });
      } else if (e.key === 'ArrowDown') {
        window.scrollBy({ top: 100, behavior: 'smooth' });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentSectionIdx, toc]);

  // Citation click handler - scroll to reference
  useEffect(() => {
    const handleCitationClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const cite = target.closest('.citation');
      if (cite) {
        // Support both data-rid and data-ref as the citation identifier
        const refId = cite.getAttribute('data-rid') ?? cite.getAttribute('data-ref');
        if (refId) {
          const refEl = document.getElementById(`ref-${refId}`);
          if (refEl) {
            e.preventDefault();
            refEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            // Highlight the reference briefly
            refEl.classList.add('bg-yellow-100');
            setTimeout(() => {
              refEl.classList.remove('bg-yellow-100');
            }, 1800);
          }
        }
      }
    };

    document.addEventListener('click', handleCitationClick);
    return () => document.removeEventListener('click', handleCitationClick);
  }, []);

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-white w-full max-w-full overflow-x-hidden">
      {/* Left Sidebar: Table of Contents */}
      <aside className="hidden lg:flex flex-col w-[260px] border-r border-slate-100 p-6 space-y-6 h-screen sticky top-0 overflow-y-auto shrink-0 bg-white custom-scrollbar">
          <div className="mb-4">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-2 title-font">
            Table of Contents
          </h3>
            <p className="text-[11px] font-semibold text-slate-700 leading-snug line-clamp-3 title-font">
            {title || 'Untitled Paper'}
          </p>
        </div>

        <nav className="flex flex-col space-y-0.5">
          {toc.map((item, idx) => (
            <button
              key={item.id}
              onClick={() => {
                setCurrentSectionIdx(idx);
                setActiveHeading(null);
                scrollToId(item.id);
              }}
              data-toc-id={`toc-${item.id}`}
              className={`toc-lnk w-full text-left text-[11px] font-semibold uppercase tracking-wider transition-all duration-200 py-2.5 px-4 rounded-xl
                ${
                  idx === currentSectionIdx && !activeHeading
                    ? 'toc-item-active bg-blue-50 text-blue-600'
                    : idx === currentSectionIdx
                    ? 'text-blue-500'
                    : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'
                }`}
              style={{ paddingLeft: `${(item.level || 1) * 0.5 + 1}rem` }}
            >
              {item.text}
            </button>
          ))}

          {/* Keyboard shortcuts */}
        <div className="hidden md:flex items-center justify-end space-x-2 text-[9px] text-slate-400">
            <span className="flex items-center">
              <span className="mr-1">←</span> Prev
            </span>
            <span className="mx-2">|</span>
            <span className="flex items-center">
              <span className="mr-1">→</span> Next
            </span>
            <span className="mx-2">|</span>
            <span className="flex items-center">
              <span className="mr-1">↑</span> Up
            </span>
            <span className="mx-2">|</span>
            <span className="flex items-center">
              <span className="mr-1">↓</span> Down
            </span>
          </div>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 bg-white border-r border-slate-100 relative">
        {/* Header */}
        <div className="p-10 lg:p-14 border-b border-slate-50 flex justify-between items-center bg-white sticky top-0 z-20">
          <div>
            <div className="flex items-center mb-2">
              <div className="flex items-center space-x-3">
                <span className="text-[9px] font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full uppercase tracking-widest">
                  {mode}
                </span>
                {fallbackSource && (
                  <span
                    className="text-[9px] font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full uppercase tracking-widest"
                  >
                    {fallbackSource.source}
                  </span>
                )}
                <a
                  href={`https://doi.org/${doi}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[9px] text-slate-400 font-medium tracking-wider uppercase hover:text-blue-600 transition-colors"
                >
                  DOI: {doi}
                </a>
              </div>
              {paperDate && (
                <span className="text-[9px] text-slate-400 font-medium tracking-wider ml-auto">
                  {paperDate}
                </span>
              )}
            </div>
              <h1
                ref={titleContainerRef}
                data-disabled-entity-groups={disabledHighlightGroupData}
                className="text-3xl font-bold text-slate-900 tracking-tight title-font"
                dangerouslySetInnerHTML={{ __html: sanitizeHtml(title || 'Untitled Paper') }}
              />
            {(paperAuthors.length > 0 || paperJournal) && (
              <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-500">
                {paperAuthors.length > 0 && (
                  <span className="truncate max-w-md" title={paperAuthors.join(', ')}>
                    {paperAuthors.slice(0, 3).join(', ')}{paperAuthors.length > 3 ? ' et al.' : ''}
                  </span>
                )}
                {paperJournal && (
                  <>
                    <span className="text-slate-300">•</span>
                    <span>{paperJournal}</span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Section Content - Continuous Scroll (HTML blob) */}
         <div
            ref={htmlContainerRef}
            id="section-content-area"
            data-disabled-entity-groups={disabledHighlightGroupData}
            className="p-10 lg:px-20 lg:py-14 min-h-screen article-prose"
            dangerouslySetInnerHTML={{ __html: sanitizeHtml(html) }}
          />
        {activeSpeciesPopup && (
          <div
            ref={speciesPopupRef}
            role="dialog"
            aria-label={`Species metadata for ${activeSpeciesPopup.species.primaryName}`}
            className="fixed z-40 w-[min(20rem,calc(100vw-1.5rem))] rounded-2xl border border-emerald-100 bg-white p-4 shadow-2xl shadow-slate-900/10"
            style={{
              top: `${activeSpeciesPopup.position.top}px`,
              left: `${activeSpeciesPopup.position.left}px`,
            }}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-emerald-600">
                  Species
                </p>
                <h3 className="mt-2 text-sm font-semibold text-slate-900 italic leading-snug">
                  {activeSpeciesPopup.species.scientificNameVerified || activeSpeciesPopup.species.primaryName}
                </h3>
                {activeSpeciesPopup.species.commonName && normalizeLookupText(activeSpeciesPopup.species.commonName) !== normalizeLookupText(activeSpeciesPopup.species.primaryName) && (
                  <p className="mt-1 text-[11px] font-medium text-slate-500 normal-case">
                    {activeSpeciesPopup.species.commonName}
                  </p>
                )}
              </div>
              <button
                type="button"
                onClick={closeSpeciesPopup}
                className="rounded-lg border border-slate-200 px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400 transition-colors hover:border-slate-300 hover:text-slate-600"
              >
                Close
              </button>
            </div>

            <div className="mt-4 space-y-2 border-t border-slate-100 pt-3 text-[10px] text-slate-600">
              {/* Taxon ID */}
              {activeSpeciesPopup.species.taxonId && (
                <div className="flex items-start gap-2">
                  <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-slate-400">Taxon</span>
                  <span className="min-w-0 break-words font-medium">{activeSpeciesPopup.species.taxonId}</span>
                </div>
              )}

              {/* Source DB and URL */}
              {(activeSpeciesPopup.species.sourceDb || activeSpeciesPopup.species.sourceUrl) && (
                <div className="flex items-start gap-2">
                  <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-slate-400">Source</span>
                  {activeSpeciesPopup.species.sourceUrl ? (
                    <a
                      href={activeSpeciesPopup.species.sourceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="min-w-0 break-words font-medium text-blue-600 hover:underline"
                    >
                      {activeSpeciesPopup.species.sourceDb || 'View'}
                    </a>
                  ) : (
                    <span className="min-w-0 break-words font-medium">{activeSpeciesPopup.species.sourceDb}</span>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
        {activeChemicalPopup && (
          <div
            ref={chemicalPopupRef}
            role="dialog"
            aria-label={`Chemical metadata for ${activeChemicalPopup.chemical.primaryName}`}
            className="fixed z-40 w-[min(20rem,calc(100vw-1.5rem))] rounded-2xl border border-blue-100 bg-white p-4 shadow-2xl shadow-slate-900/10"
            style={{
              top: `${activeChemicalPopup.position.top}px`,
              left: `${activeChemicalPopup.position.left}px`,
            }}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-blue-600">
                  Chemical
                </p>
                <h3 className="mt-2 text-sm font-semibold text-slate-900 leading-snug">
                  {activeChemicalPopup.chemical.primaryName}
                </h3>
                {activeChemicalPopup.chemical.preferredName && normalizeLookupText(activeChemicalPopup.chemical.preferredName) !== normalizeLookupText(activeChemicalPopup.chemical.primaryName) && (
                  <p className="mt-1 text-[11px] font-medium text-slate-500 normal-case">
                    {activeChemicalPopup.chemical.preferredName}
                  </p>
                )}
                {activeChemicalPopup.chemical.synonyms && activeChemicalPopup.chemical.synonyms.length > 0 && (
                  <div className="mt-1 text-[10px] font-medium text-slate-400 normal-case">
                    Also: {isExpandedChemical ? (
                      <>
                        {activeChemicalPopup.chemical.synonyms.join(', ')}
                        <button
                          type="button"
                          onClick={() => toggleExpandedChemical(activeChemicalPopup.chemical.primaryName)}
                          className="ml-1 font-semibold text-blue-600 hover:underline"
                        >
                          Show less
                        </button>
                      </>
                    ) : (
                      <>
                        {activeChemicalPopup.chemical.synonyms.slice(0, 3).join(', ')}
                        {activeChemicalPopup.chemical.synonyms.length > 3 && (
                          <button
                            type="button"
                            onClick={() => toggleExpandedChemical(activeChemicalPopup.chemical.primaryName)}
                            className="ml-1 font-semibold text-blue-600 hover:underline"
                          >
                            +{activeChemicalPopup.chemical.synonyms.length - 3} more
                          </button>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
              <button
                type="button"
                onClick={closeChemicalPopup}
                className="rounded-lg border border-slate-200 px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400 transition-colors hover:border-slate-300 hover:text-slate-600"
              >
                Close
              </button>
            </div>

            {/* Molecular Structure */}
            <div className="mt-3 min-h-[19rem] rounded-xl border border-slate-100 bg-slate-50/70 px-3 py-4">
              {activeChemicalPopup.chemical.smiles ? (
                <div className="relative flex min-h-[17rem] items-center justify-center">
                  {isRenderingChemicalStructure && (
                    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 rounded-lg bg-white/75 backdrop-blur-[1px]">
                      <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-200 border-t-slate-500" />
                      <p className="text-xs font-medium text-slate-500">Rendering structure…</p>
                    </div>
                  )}
                  {!chemicalStructureError ? (
                    <svg
                      ref={chemicalStructureSvgRef}
                      className="h-auto max-h-[18rem] w-full max-w-[24rem]"
                      viewBox="0 0 400 300"
                      aria-label={`${activeChemicalPopup.chemical.primaryName} molecular structure`}
                    />
                  ) : (
                    <div className="text-xs text-slate-400">No structure</div>
                  )}
                </div>
              ) : (
                <div className="flex min-h-[17rem] items-center justify-center text-xs text-slate-400">No structure</div>
              )}
            </div>

            <div className="mt-4 space-y-2 border-t border-slate-100 pt-3 text-[10px] text-slate-600">
              {/* Molecular Formula */}
              {activeChemicalPopup.chemical.molecularFormula && (
                <div className="flex items-start gap-2">
                  <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-slate-400">Formula</span>
                  <span className="min-w-0 break-words font-mono font-medium">{activeChemicalPopup.chemical.molecularFormula}</span>
                </div>
              )}

              {/* InChIKey */}
              {activeChemicalPopup.chemical.inchikey && (
                <div className="flex items-start gap-2">
                  <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-slate-400">InChIKey</span>
                  <span className="min-w-0 break-words font-mono font-medium break-all">{activeChemicalPopup.chemical.inchikey}</span>
                </div>
              )}

              {/* Source DB and URL */}
              {(activeChemicalPopup.chemical.sourceDb || activeChemicalPopup.chemical.sourceUrl) && (
                <div className="flex items-start gap-2">
                  <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-slate-400">Source</span>
                  {activeChemicalPopup.chemical.sourceUrl ? (
                    <a
                      href={activeChemicalPopup.chemical.sourceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="min-w-0 break-words font-medium text-blue-600 hover:underline"
                    >
                      {activeChemicalPopup.chemical.sourceDb || 'View'}
                    </a>
                  ) : (
                    <span className="min-w-0 break-words font-medium">{activeChemicalPopup.chemical.sourceDb}</span>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
        {isFetchingFallback && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-slate-200 border-t-emerald-600 mx-auto mb-3" />
              <p className="text-xs text-slate-400">Fetching abstract from alternative sources...</p>
            </div>
          </div>
        )}
      </main>

      {/* Right Sidebar: Entity Groups */}
      <aside className="w-full lg:w-[380px] p-8 space-y-10 bg-slate-50/20 h-screen sticky top-0 overflow-y-auto custom-scrollbar shrink-0 relative z-30">
        {/* Find Key Terms Button */}
        <button
          onClick={() => {
            if (onExtract && !isExtracting && !isExtracted) onExtract();
          }}
          disabled={isExtracted || isExtracting}
          className={`w-full px-6 py-3 rounded-xl font-bold transition-all flex items-center justify-center space-x-2 text-[11px] uppercase tracking-widest shadow-lg ${
            isExtracted 
              ? 'bg-slate-200 text-slate-500 cursor-not-allowed' 
              : isExtracting
              ? 'bg-blue-100 text-blue-400 cursor-wait animate-pulse'
              : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-100 cursor-pointer'
          }`}
        >
          {isExtracting ? (
            <div className="animate-spin rounded-full h-3 w-3 border-2 border-blue-400 border-t-blue-600 mr-2" />
          ) : (
            <PencilSimple size={16} weight="bold" />
          )}
          <span>
            {isExtracted ? 'Entities Extracted' : isExtracting ? 'Extracting Terms...' : 'Find Key Terms'}
          </span>
        </button>

        {/* Export CSV */}
        {isExtracted && entities.length > 0 && (
          <button
            onClick={() => {
              const escape = (val: string | number | undefined | null) => {
                if (val == null) return '';
                const str = String(val);
                return str.includes(',') || str.includes('"') || str.includes('\n')
                  ? `"${str.replace(/"/g, '""')}"`
                  : str;
              };
              // Build grouped data from existing grouping logic
              const grouped = groupedEntityMap;
              // Sort groups alphabetically (already alphabetical by label names)
              const sortedLabels = Object.keys(grouped).sort((a, b) => a.localeCompare(b));
              const rows: string[] = [];
              for (const label of sortedLabels) {
                const items = [...grouped[label]].sort((a, b) => a.text.localeCompare(b.text));
                for (const item of items) {
                  rows.push(
                    [label, item.text, String(item.count), item.aliases.slice(0, 5).join('; ')].map(escape).join(',')
                  );
                }
              }
              const csv = `# DOI: ${doi}\nType,Name,Count,Variants\n${rows.join('\n')}`;
              const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `entities-${doi.replace(/[^a-zA-Z0-9.-]/g, '_')}.csv`;
              a.click();
              URL.revokeObjectURL(url);
            }}
            className="w-full px-6 py-3 rounded-xl font-bold transition-all flex items-center justify-center space-x-2 text-[11px] uppercase tracking-widest shadow-lg bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-100 cursor-pointer"
          >
            <DownloadSimple size={16} weight="bold" />
            <span>Export CSV</span>
          </button>
        )}

        {extractionError && (
          <div className="p-4 bg-red-50 border border-red-100 rounded-xl">
            <p className="text-[10px] text-red-600 font-bold uppercase tracking-wider mb-1">Extraction Error</p>
            <p className="text-[11px] text-red-500 leading-relaxed font-medium">{extractionError}</p>
          </div>
        )}

        <div className="space-y-6">
          <div className="border-b border-slate-100 pb-4">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] title-font">
              Extracted Mentions
            </h3>
            <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
              Expand entity groups to inspect extracted mentions, or switch highlights on and off in the paper view.
            </p>
          </div>

          {isExtracted && (
            <div className="relative">
              <input
                type="text"
                value={entitySearch}
                onChange={(e) => setEntitySearch(e.target.value)}
                placeholder="Search terms..."
                className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-[11px] font-medium text-slate-700 placeholder:text-slate-400 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
              />
            </div>
          )}

          {!isExtracted ? (
            <div className="text-center py-10 bg-slate-100/50 rounded-2xl border border-dashed border-slate-200">
              <p className="text-[10px] text-slate-400 uppercase tracking-widest italic font-medium">
                Pending analysis
              </p>
            </div>
          ) : (
            <div className="space-y-3 pr-2">
              {!hasVisibleEntityMatches && normalizedEntitySearch && (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-white px-4 py-5 text-center">
                  <p className="text-[10px] font-medium uppercase tracking-widest text-slate-400 italic">
                    No matching entities
                  </p>
                </div>
              )}

              {visibleGroupedEntities.map((group, idx) => {
                const isExpanded = expandedGroups[group.label];
                const isHighlightEnabled = enabledHighlightGroups[group.label];
                const panelId = getEntityGroupPanelId(group.label);
                const accentColor = `var(${group.accentVar})`;
                const emptyMessage = normalizedEntitySearch && group.termCount > 0
                  ? 'No matching mentions in this group.'
                  : 'No extracted mentions in this group yet.';

                return (
                  <section
                    key={group.label}
                    className="animate-fade-in overflow-hidden rounded-2xl border border-slate-200/80 bg-white shadow-[0_16px_35px_-28px_rgba(15,23,42,0.32)]"
                    style={{ animationDelay: `${idx * 0.05}s` }}
                  >
                    <div className="flex items-center gap-2 border-b border-slate-100 px-3 py-3">
                      <button
                        type="button"
                        aria-expanded={isExpanded}
                        aria-controls={panelId}
                        onClick={() => toggleEntityGroup(group.label)}
                        className="flex flex-1 items-center gap-3 rounded-xl px-2 py-2 text-left transition-colors hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-200"
                      >
                        <span
                          className="h-3.5 w-3.5 shrink-0 rounded-[0.45rem] border border-white/70 shadow-sm"
                          style={{ backgroundColor: accentColor }}
                          aria-hidden="true"
                        />
                        <span className="min-w-0 flex-1">
                          <span className="block text-[10px] font-bold uppercase tracking-[0.18em] text-slate-700">
                            {group.label}
                          </span>
                          <span className="mt-1 block text-[10px] font-medium text-slate-400">
                            {group.termCount > 0
                              ? `${group.termCount} term${group.termCount === 1 ? '' : 's'}`
                              : 'No extracted terms'}
                          </span>
                        </span>
                        <span
                          className="rounded-full px-2.5 py-1 text-[10px] font-bold"
                          style={{ border: `1px solid ${accentColor}`, color: accentColor }}
                        >
                          {group.totalCount}
                        </span>
                        <span
                          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-slate-200 text-slate-400 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
                          aria-hidden="true"
                        >
                          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        </span>
                      </button>

                      <label className="flex shrink-0 cursor-pointer items-center gap-2 rounded-xl border border-slate-200 px-3 py-2 hover:border-slate-300">
                        <input
                          type="checkbox"
                          checked={isHighlightEnabled}
                          onChange={() => toggleHighlightGroup(group.label)}
                          aria-label={`${isHighlightEnabled ? 'Disable' : 'Enable'} ${group.label.toLowerCase()} highlights in the paper view`}
                          className="sr-only peer"
                        />
                        <span className="text-[9px] font-bold uppercase tracking-[0.18em] text-slate-400">
                          {isHighlightEnabled ? 'On' : 'Off'}
                        </span>
                        <span
                          aria-hidden="true"
                          className="relative h-5 w-9 rounded-full bg-slate-200 transition-colors peer-checked:bg-slate-900/80"
                        >
                          <span className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform peer-checked:translate-x-4" />
                        </span>
                      </label>
                    </div>

                    <div id={panelId} hidden={!isExpanded} className="px-3 pb-3 pt-3">
                      {group.visibleItems.length === 0 ? (
                        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-[10px] font-medium text-slate-400">
                          {emptyMessage}
                        </div>
                      ) : (
                        <div className="space-y-1.5">
                          {group.visibleItems.map((ent, eIdx) => {
                            const expandKey = `${group.label}-${ent.text.toLowerCase()}-${eIdx}`;
                            const isChemicalRowExpanded = expandedChemical === expandKey;
                            const hasChemicalMeta = isChemicalLikeLabel(group.label) && (
                              ent.molecular_formula || ent.inchikey || ent.smiles || ent.source_db
                            );

                            if (isChemicalLikeLabel(group.label) && hasChemicalMeta) {
                              return (
                                <div key={eIdx} className="bg-white border border-slate-100 rounded-xl overflow-hidden transition-all hover:border-blue-200">
                                  <button
                                    type="button"
                                    onClick={() => setExpandedChemical(isChemicalRowExpanded ? null : expandKey)}
                                    className="w-full flex items-center justify-between py-2.5 px-4 text-left cursor-pointer"
                                  >
                                    <div className="min-w-0 pr-2">
                                      <div className="text-[10px] text-slate-600 font-bold truncate uppercase">
                                        {ent.text}
                                      </div>
                                      {ent.molecular_formula && (
                                        <div className="text-[9px] text-slate-400 font-mono mt-0.5 truncate normal-case">
                                          {ent.molecular_formula}
                                        </div>
                                      )}
                                    </div>
                                    <div className="flex items-center shrink-0 gap-2">
                                      <span className="text-[10px] font-extrabold text-blue-600">
                                        {ent.count}
                                      </span>
                                      <span className={`text-slate-400 transition-transform duration-200 ${isChemicalRowExpanded ? 'rotate-180' : ''}`}>
                                        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                                          <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                        </svg>
                                      </span>
                                    </div>
                                  </button>

                                  <div className={`overflow-hidden transition-all duration-200 ${isChemicalRowExpanded ? 'max-h-48' : 'max-h-0'}`}>
                                    <div className="px-4 pb-3 pt-1 border-t border-slate-50 space-y-1.5">
                                      {ent.preferred_name && ent.preferred_name !== ent.text && (
                                        <div className="flex items-start gap-2">
                                          <span className="text-[9px] font-semibold text-slate-400 shrink-0 w-16">Name</span>
                                          <span className="text-[9px] text-slate-600 font-medium truncate">{ent.preferred_name}</span>
                                        </div>
                                      )}
                                      {ent.molecular_formula && (
                                        <div className="flex items-start gap-2">
                                          <span className="text-[9px] font-semibold text-slate-400 shrink-0 w-16">Formula</span>
                                          <span className="text-[9px] text-slate-600 font-mono font-medium">{ent.molecular_formula}</span>
                                        </div>
                                      )}
                                      {ent.inchikey && (
                                        <div className="flex items-start gap-2">
                                          <span className="text-[9px] font-semibold text-slate-400 shrink-0 w-16">InChIKey</span>
                                          <span className="text-[9px] text-slate-600 font-mono font-medium break-all">{ent.inchikey}</span>
                                        </div>
                                      )}
                                      {ent.smiles && (
                                        <div className="flex items-start gap-2">
                                          <span className="text-[9px] font-semibold text-slate-400 shrink-0 w-16">SMILES</span>
                                          <span className="text-[9px] text-slate-600 font-mono font-medium break-all">{ent.smiles}</span>
                                        </div>
                                      )}
                                      {ent.source_db && (
                                        <div className="flex items-start gap-2">
                                          <span className="text-[9px] font-semibold text-slate-400 shrink-0 w-16">Source</span>
                                          {ent.source_url ? (
                                            <a
                                              href={ent.source_url}
                                              target="_blank"
                                              rel="noopener noreferrer"
                                              className="text-[9px] text-blue-600 hover:underline font-medium truncate"
                                            >
                                              {ent.source_db}
                                            </a>
                                          ) : (
                                            <span className="text-[9px] text-slate-600 font-medium">{ent.source_db}</span>
                                          )}
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              );
                            }

                            return (
                              <div
                                key={eIdx}
                                className="flex justify-between items-center py-2.5 px-4 bg-white border border-slate-100 rounded-xl transition-all hover:border-blue-100 group/item"
                              >
                                <div className="min-w-0 pr-2">
                                  <div className={`text-[10px] text-slate-600 font-bold truncate ${group.label === 'SPECIES' ? '' : 'uppercase'}`}>
                                    {group.label === 'SPECIES' ? getSpeciesPrimaryName({
                                      text: ent.text,
                                      label: 'SPECIES',
                                      score: 1,
                                      accepted_scientific_name: ent.text,
                                      scientific_name_verified: ent.text,
                                      canonical: ent.text,
                                    } as Entity) : ent.text}
                                  </div>
                                  {ent.subtitle && (
                                    <div className="text-[10px] text-slate-400 truncate mt-0.5 normal-case font-medium">
                                      {ent.subtitle}
                                    </div>
                                  )}
                                </div>
                                <span className="text-[10px] font-extrabold text-blue-600 shrink-0">
                                  {ent.count}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </section>
                );
              })}
            </div>
          )}
        </div>
      </aside>

      <style>{`
        .font-display {
          font-family: 'Inter', sans-serif !important;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.4s ease-out forwards;
        }
        .toc-lnk:focus-visible {
          outline: 2px solid #2563eb;
          outline-offset: 2px;
        }
        .scroll-mt-20 {
          scroll-margin-top: 5rem;
        }
        ${ENTITY_GROUP_ORDER.map((label) => `
          [data-disabled-entity-groups~="${label}"] :is(${ENTITY_GROUP_CONFIG[label].highlightSelector}) {
            background-color: transparent !important;
            box-shadow: none !important;
            color: inherit !important;
            font-weight: inherit !important;
            padding: 0 !important;
          }
        `).join('')}
      `}</style>
    </div>
  );
};

export default PaperViewer;
