import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { sanitizeHtml, formatTextWithFormatting } from '../../utils/sanitize';
import { ListMagnifyingGlass, Chats, Download, PencilSimple, Table } from '@phosphor-icons/react';
import type { Entity, TocItem } from '../../types';
import SmilesDrawer from 'smiles-drawer';
import { KnowledgeGraph } from './KnowledgeGraph';

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
    highlightSelector: '.ent-analytical-technique, mark.ner-analytical-technique, .ent-isolation-method, mark.ner-isolation-method',
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

const createInitialExpandedGroups = () => {
  return ENTITY_GROUP_ORDER.reduce<Record<EntityGroupLabel, boolean>>((acc, label) => {
    acc[label] = false;
    return acc;
  }, {} as Record<EntityGroupLabel, boolean>);
};

const createInitialEnabledHighlightGroups = () => {
  return ENTITY_GROUP_ORDER.reduce<Record<EntityGroupLabel, boolean>>((acc, label) => {
    acc[label] = true;
    return acc;
  }, {} as Record<EntityGroupLabel, boolean>);
};

const getEntityGroupToken = (label: EntityGroupLabel) => label.toLowerCase().replace(/[^a-z0-9]+/g, '-');



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
  paperIdentifier?: {
    type: 'doi' | 'pmcid' | 'pmid';
    value: string;
    href: string;
  };
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
  canUsePdfActions?: boolean;
  isDownloadingPdf?: boolean;
  isUploadingToRag?: boolean;
  isAddingToAnalyse?: boolean;
  pdfActionError?: string | null;
  analyseActionError?: string | null;
  onDownloadPdf?: () => void;
  onSendPdfToRag?: () => void;
  onAddToAnalyse?: () => void;
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
   paperIdentifier,
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
  canUsePdfActions = false,
  isDownloadingPdf = false,
  isUploadingToRag = false,
  isAddingToAnalyse = false,
  pdfActionError = null,
  analyseActionError = null,
  onDownloadPdf,
  onSendPdfToRag,
  onAddToAnalyse,
  onExtract,
}) => {
  const identifierValue = paperIdentifier?.value || 'paper';
  const pdfToolbarError = analyseActionError || pdfActionError;

  const [activeHeading, setActiveHeading] = useState<string | null>(null);
  const [currentSectionIdx, setCurrentSectionIdx] = useState(0);
  const [expandedChemical, setExpandedChemical] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Record<EntityGroupLabel, boolean>>(() => createInitialExpandedGroups());
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

  const toggleAllHighlightGroups = useCallback(() => {
    setEnabledHighlightGroups((current) => {
      const allEnabled = ENTITY_GROUP_ORDER.every(label => current[label]);
      const nextState = {} as Record<EntityGroupLabel, boolean>;
      ENTITY_GROUP_ORDER.forEach(label => {
        nextState[label] = !allEnabled;
      });
      return nextState;
    });
  }, []);

  const activeGroupsCount = Object.values(enabledHighlightGroups).filter(Boolean).length;
  const isAllEnabled = activeGroupsCount === ENTITY_GROUP_ORDER.length;
  const isNoneEnabled = activeGroupsCount === 0;

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

  const visibleGroupedEntities = useMemo(() => {
    return ENTITY_GROUP_ORDER.map((label) => {
      const items = groupedEntityMap[label] ?? [];
      return {
        label,
        items,
        visibleItems: items,
        totalCount: items.reduce((sum, item) => sum + item.count, 0),
        termCount: items.length,
        ...ENTITY_GROUP_CONFIG[label],
      };
    });
  }, [groupedEntityMap]);

  const disabledHighlightGroups = useMemo(() => {
    return ENTITY_GROUP_ORDER.filter((label) => !enabledHighlightGroups[label]);
  }, [enabledHighlightGroups]);

  const disabledHighlightGroupData = disabledHighlightGroups.map(getEntityGroupToken).join(' ') || undefined;
  
  const resetReaderUiState = useCallback(() => {
    setExpandedChemical(null);
    setExpandedGroups(createInitialExpandedGroups());
    setEnabledHighlightGroups(createInitialEnabledHighlightGroups());
    closeSpeciesPopup();
    closeChemicalPopup();
  }, [isExtracted, groupedEntityMap, closeSpeciesPopup, closeChemicalPopup]);

  useEffect(() => {
    const resetTimer = window.setTimeout(() => {
      resetReaderUiState();
    }, 0);

    return () => window.clearTimeout(resetTimer);
  }, [identifierValue, html, resetReaderUiState]);

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

  // Setup scroll spy for section headings
  useEffect(() => {
    if (!htmlContainerRef.current) return;
    const container = htmlContainerRef.current;
    const headingNodes = container.querySelectorAll('h3.article-h3');
    if (headingNodes.length === 0) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.intersectionRatio > 0) {
            const id = entry.target.id;
            // Find matching TOC item by ID
            const tocItem = toc.find(item => item.id === id);
            if (tocItem) {
              const idx = toc.indexOf(tocItem);
              if (idx >= 0) setCurrentSectionIdx(idx);
            }
            setActiveHeading(id);
          }
        });
      },
      { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
    );

    headingNodes.forEach((el: Element) => observerRef.current?.observe(el as Element));

    return () => {
      observerRef.current?.disconnect();
    };
  }, [html, toc]);

  // Scroll to element by ID
  const scrollToId = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Citation click handler - REMOVED (references section no longer displayed)

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-white w-full max-w-full overflow-x-hidden">
      {/* Left Sidebar: Table of Contents */}
<aside className="hidden lg:flex flex-col w-[260px] border-r border-slate-100 p-6 space-y-6 h-screen sticky top-0 overflow-y-auto shrink-0 bg-white custom-scrollbar">
          <div className="mb-4">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] title-font">
              Sections
            </h3>
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
                className={`toc-lnk w-full text-left text-[11px] font-semibold uppercase tracking-wider transition-all duration-200 py-2.5 px-4 rounded-xl relative
                  ${
                    idx === currentSectionIdx && !activeHeading
                      ? 'toc-item-active bg-blue-50 text-blue-600'
                      : idx === currentSectionIdx
                      ? 'text-blue-500'
                      : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'
                  }`}
                style={{ paddingLeft: `${(item.level || 1) * 0.5 + 1}rem` }}
              >
                {idx === currentSectionIdx && !activeHeading && (
                  <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-blue-600 rounded-r" />
                )}
                {item.text}
              </button>
            ))}
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
                      className={`text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-widest ${
                        fallbackSource.source.toLowerCase().includes('openalex')
                          ? 'bg-slate-100 text-slate-700'
                          : fallbackSource.source.toLowerCase().includes('semantic')
                          ? 'bg-yellow-50 text-yellow-700'
                          : fallbackSource.source.toLowerCase().includes('europe')
                          ? 'bg-emerald-50 text-emerald-600'
                          : 'bg-slate-100 text-slate-700'
                      }`}
                    >
                      {fallbackSource.source}
                    </span>
                  )}
                 {paperIdentifier && (
                  <a
                    href={paperIdentifier.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[9px] text-slate-400 font-medium tracking-wider uppercase hover:text-blue-600 transition-colors"
                  >
                    {paperIdentifier.type.toUpperCase()}: {paperIdentifier.value}
                  </a>
                )}
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
                dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(title) || 'Untitled Paper' }}
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
            {canUsePdfActions && (
              <>
                <div className="mt-4 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={onDownloadPdf}
                    disabled={isDownloadingPdf}
                    title="Download"
                    className={`inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500 shadow-sm transition-colors hover:border-slate-300 hover:text-slate-700 ${
                      isDownloadingPdf ? 'cursor-wait opacity-60' : ''
                    }`}
                  >
                    {isDownloadingPdf ? (
                      <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600" />
                    ) : (
                      <Download size={18} weight="bold" />
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={onAddToAnalyse}
                    disabled={isAddingToAnalyse}
                    title="Analyse"
                    className={`inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500 shadow-sm transition-colors hover:border-slate-300 hover:text-slate-700 ${
                      isAddingToAnalyse ? 'cursor-wait opacity-60' : ''
                    }`}
                  >
                    {isAddingToAnalyse ? (
                      <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600" />
                    ) : (
                      <ListMagnifyingGlass size={18} weight="bold" />
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={onSendPdfToRag}
                    disabled={isUploadingToRag}
                    title="Chat"
                    className={`inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500 shadow-sm transition-colors hover:border-slate-300 hover:text-slate-700 ${
                      isUploadingToRag ? 'cursor-wait opacity-60' : ''
                    }`}
                  >
                    {isUploadingToRag ? (
                      <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600" />
                    ) : (
                      <Chats size={18} weight="bold" />
                    )}
                  </button>
                </div>

                {pdfToolbarError && (
                  <p className="mt-2 text-xs text-red-500">{pdfToolbarError}</p>
                )}
              </>
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
              const csv = `# ${paperIdentifier?.type?.toUpperCase() || 'PAPER'}: ${identifierValue}\nType,Name,Count,Variants\n${rows.join('\n')}`;
              const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `entities-${identifierValue.replace(/[^a-zA-Z0-9.-]/g, '_')}.csv`;
              a.click();
              URL.revokeObjectURL(url);
            }}
            className="w-full px-6 py-3 rounded-xl font-bold transition-all flex items-center justify-center space-x-2 text-[11px] uppercase tracking-widest shadow-lg bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-100 cursor-pointer"
          >
            <Table size={16} weight="bold" />
            <span>Export Entities</span>
          </button>
        )}

        {extractionError && (
          <div className="p-4 bg-red-50 border border-red-100 rounded-xl">
            <p className="text-[10px] text-red-600 font-bold uppercase tracking-wider mb-1">Extraction Error</p>
            <p className="text-[11px] text-red-500 leading-relaxed font-medium">{extractionError}</p>
          </div>
        )}

        <div className="flex flex-col flex-1 mt-2">
          {/* Header & Master Toggle */}
          <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-4">
            <div className="flex items-center gap-3">
              <span className="text-[14px] font-semibold text-slate-900 font-display">Entity Index</span>
            </div>
            
            <div 
              className="flex items-center gap-2 cursor-pointer select-none group" 
              onClick={toggleAllHighlightGroups}
            >
              <div className={`w-[16px] h-[16px] rounded border-[1.5px] flex items-center justify-center transition-all ${isAllEnabled ? 'bg-slate-900 border-slate-900' : isNoneEnabled ? 'bg-white border-slate-300 group-hover:border-slate-400' : 'bg-white border-slate-400'}`}>
                {isAllEnabled && (
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                    <polyline points="1.5,5 4,7.5 8.5,2.5" stroke="#fff" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )}
                {!isAllEnabled && !isNoneEnabled && (
                  <div className="w-[8px] h-[1.5px] bg-slate-500 rounded-sm"></div>
                )}
              </div>
              <span className="text-[10px] font-semibold text-slate-500 tracking-wide font-display">{isAllEnabled ? 'All' : isNoneEnabled ? 'None' : `${activeGroupsCount}`}</span>
            </div>
          </div>

            <div className="flex-1 overflow-y-auto pr-2 -mr-2">

              <div className="flex flex-col bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
                {visibleGroupedEntities.map((group) => {
                  const isExpanded = expandedGroups[group.label];
                  const isHighlightEnabled = enabledHighlightGroups[group.label];
                  const accentColor = `var(${group.accentVar})`;

                  return (
                    <div 
                      key={group.label} 
                      className="relative border-b-[0.5px] border-slate-200 last:border-0" 
                      style={{ borderLeft: isExpanded ? `2.5px solid ${accentColor}` : '2.5px solid transparent' }}
                    >
                      {/* Entity Row (.e-row) */}
                      <div 
                        className={`flex items-center gap-3 py-2.5 px-3 cursor-pointer transition-colors hover:bg-slate-50 ${isHighlightEnabled ? '' : 'opacity-50 grayscale-[0.5]'}`}
                        onClick={() => toggleHighlightGroup(group.label)}
                      >
                        <div 
                          className="w-[3px] h-8 rounded-sm shrink-0" 
                          style={{ backgroundColor: accentColor, opacity: isHighlightEnabled ? (group.termCount === 0 ? 0.3 : 1) : 0.15 }}
                        />
                        <div className="flex-1 flex flex-col gap-[2px] min-w-0">
                          <span 
                            className="text-[11px] font-semibold text-slate-800 font-display truncate capitalize"
                            style={{ color: isExpanded ? accentColor : undefined }}
                          >
                            {group.label.toLowerCase()}
                          </span>
                          {group.termCount > 0 && (
                            <div 
                              className="h-2 w-full rounded-[2px] transition-opacity"
                              style={{ 
                                backgroundColor: `rgb(var(${group.accentVar}-rgb) / 0.13)`, 
                                opacity: isHighlightEnabled ? 1 : 0.1
                              }}
                            />
                          )}
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <span 
                            className="text-[10px] font-mono text-right min-w-[1.5rem]"
                            style={{ color: group.termCount === 0 ? '#cbd5e1' : accentColor, opacity: isHighlightEnabled ? 1 : 0.2 }}
                          >
                            {group.termCount || '—'}
                          </span>
                          {group.termCount > 0 && (
                            <button 
                              type="button"
                              className={`text-[9px] text-slate-300 transition-transform duration-200 px-1.5 py-1 hover:text-slate-500 ${isExpanded ? 'rotate-90' : ''}`}
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleEntityGroup(group.label);
                              }}
                            >
                              ▶
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Values Expansion (.e-values-wrap) */}
                      <div className={`grid transition-[grid-template-rows] duration-300 ease-in-out bg-white ${isExpanded && group.visibleItems.length > 0 ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}`}>
                        <div className="overflow-hidden min-h-0">
                          <div className="flex flex-col py-1 border-t-[0.5px] border-slate-100">
                            {group.visibleItems.map((ent, eIdx) => {
                              const expandKey = `${group.label}-${ent.text.toLowerCase()}-${eIdx}`;
                              const isChemicalRowExpanded = expandedChemical === expandKey;
                              const hasChemicalMeta = isChemicalLikeLabel(group.label) && (
                                ent.molecular_formula || ent.inchikey || ent.smiles || ent.source_db
                              );
                              const displayName = group.label === 'SPECIES' ? getSpeciesPrimaryName({
                                text: ent.text,
                                label: 'SPECIES',
                                score: 1,
                                accepted_scientific_name: ent.text,
                                scientific_name_verified: ent.text,
                                canonical: ent.text,
                              } as Entity) : ent.text;

                              return (
                                <div key={eIdx} className="flex flex-col border-b-[0.5px] border-slate-50 last:border-0">
                                  {/* Value Row (.v-row) */}
                                  <div 
                                    className={`flex items-center gap-2 py-1.5 pr-3 pl-8 transition-colors ${hasChemicalMeta ? 'cursor-pointer hover:bg-slate-50' : ''}`}
                                    onClick={(e) => {
                                      if (hasChemicalMeta) {
                                        e.stopPropagation();
                                        setExpandedChemical(isChemicalRowExpanded ? null : expandKey);
                                      }
                                    }}
                                  >
                                    <div className="w-[6px] h-[6px] rounded-full shrink-0" style={{ backgroundColor: accentColor }} />
                                    <div className="flex-1 min-w-0 pr-2">
                                      <span 
                                        className={`text-[11px] font-display block truncate ${isChemicalRowExpanded ? 'font-semibold' : 'text-slate-600'}`} 
                                        style={{ color: isChemicalRowExpanded ? accentColor : undefined }}
                                      >
                                        {displayName}
                                      </span>
                                    </div>
                                    <span className="text-[9px] font-mono text-slate-900 font-semibold">{ent.count}</span>
                                  </div>
                                  
                                  {/* Inline Chemical Meta */}
                                  {hasChemicalMeta && (
                                    <div className={`grid transition-[grid-template-rows] duration-200 bg-slate-50/50 ${isChemicalRowExpanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}`}>
                                      <div className="overflow-hidden min-h-0">
                                        <div className="pl-[2.6rem] pr-4 pb-2 pt-1.5 space-y-1.5 border-t border-slate-100/50">
                                        {ent.preferred_name && ent.preferred_name !== ent.text && (
                                          <div className="flex items-start gap-2">
                                            <span className="text-[9px] font-semibold text-slate-400 shrink-0 w-12">Name</span>
                                            <span className="text-[9px] text-slate-600 font-medium truncate">{ent.preferred_name}</span>
                                          </div>
                                        )}
                                        {ent.molecular_formula && (
                                          <div className="flex items-start gap-2">
                                            <span className="text-[9px] font-semibold text-slate-400 shrink-0 w-12">Formula</span>
                                            <span className="text-[9px] text-slate-600 font-mono font-medium">{ent.molecular_formula}</span>
                                          </div>
                                        )}
                                        {ent.inchikey && (
                                          <div className="flex items-start gap-2">
                                            <span className="text-[9px] font-semibold text-slate-400 shrink-0 w-12">InChIKey</span>
                                            <span className="text-[9px] text-slate-600 font-mono font-medium break-all">{ent.inchikey}</span>
                                          </div>
                                        )}
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          
            <KnowledgeGraph 
              entities={entities} 
              paperIdentifier={paperIdentifier}
              entityConfig={ENTITY_GROUP_CONFIG}
            />
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
          [data-disabled-entity-groups~="${getEntityGroupToken(label)}"] :is(${ENTITY_GROUP_CONFIG[label].highlightSelector}) {
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
