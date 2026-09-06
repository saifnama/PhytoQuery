import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { sanitizeHtml, formatTextWithFormatting } from '../../utils/sanitize';
import { downloadGraphHtml } from '../../utils/exportGraphHtml';
import { Sparkle, ListBullets, Graph, DotsThreeVertical, DownloadSimple, ListMagnifyingGlass, Chats, Eye, EyeSlash, LockSimpleOpen, Article, CaretDown, CaretUp, SpinnerGap, X } from '@phosphor-icons/react';
import type { Entity, TocItem } from '../../types';
import SmilesDrawer from 'smiles-drawer';
import { KnowledgeGraph } from './KnowledgeGraph';

const SPECIES_SELECTOR = '.ent-species, mark.ner-species';
const CHEMICAL_SELECTOR = '.ent-chemical, mark.ner-chemical';
const SPECIES_POPUP_WIDTH = 320;
const SPECIES_POPUP_ESTIMATED_HEIGHT = 160;

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
    acc[label] = label === 'CHEMICAL' || label === 'SPECIES';
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

const getAnchorRect = (
  anchor: HTMLElement,
  fallbackPoint?: { x: number; y: number }
): { top: number; bottom: number; left: number; right: number; width: number; height: number } => {
  if (anchor && typeof anchor.getBoundingClientRect === 'function') {
    const r = anchor.getBoundingClientRect();
    if (r.width > 0 || r.height > 0) {
      return r;
    }
    const clientRects = anchor.getClientRects();
    if (clientRects && clientRects.length > 0) {
      for (let i = 0; i < clientRects.length; i++) {
        const cr = clientRects[i];
        if (cr.width > 0 || cr.height > 0) {
          return cr;
        }
      }
    }
    if (anchor.parentElement) {
      const pr = anchor.parentElement.getBoundingClientRect();
      if (pr.width > 0 || pr.height > 0) {
        return pr;
      }
    }
  }
  if (fallbackPoint && fallbackPoint.x > 0 && fallbackPoint.y > 0) {
    return {
      top: fallbackPoint.y - 10,
      bottom: fallbackPoint.y + 10,
      left: fallbackPoint.x - 20,
      right: fallbackPoint.x + 20,
      width: 40,
      height: 20,
    };
  }
  return { top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 };
};

const getSpeciesPopupPosition = (
  anchor: HTMLElement,
  fallbackPoint?: { x: number; y: number },
  previousPos?: { top: number; left: number }
): { top: number; left: number } => {
  const rect = getAnchorRect(anchor, fallbackPoint);
  const margin = 12;
  const popupWidth = SPECIES_POPUP_WIDTH;

  if (rect.width === 0 && rect.height === 0 && rect.top === 0 && rect.left === 0) {
    if (previousPos) return previousPos;
  }

  // Always prefer placing directly BELOW the clicked entity word
  let top = rect.bottom + 8;

  // Only flip above if there is very little space below AND ample space above
  const spaceBelow = window.innerHeight - rect.bottom - margin;
  if (spaceBelow < 140 && rect.top - margin > 180) {
    top = Math.max(margin, rect.top - SPECIES_POPUP_ESTIMATED_HEIGHT - 8);
  } else if (top > window.innerHeight - 80) {
    top = Math.max(margin, window.innerHeight - 80);
  }

  const centeredLeft = rect.left + rect.width / 2 - popupWidth / 2;
  const left = Math.max(
    margin,
    Math.min(centeredLeft, window.innerWidth - popupWidth - margin)
  );

  return { top, left };
};

const getChemicalPopupPosition = (
  anchor: HTMLElement,
  fallbackPoint?: { x: number; y: number },
  previousPos?: { top: number; left: number }
): { top: number; left: number } => {
  const rect = getAnchorRect(anchor, fallbackPoint);
  const margin = 12;
  const popupWidth = 320;

  if (rect.width === 0 && rect.height === 0 && rect.top === 0 && rect.left === 0) {
    if (previousPos) return previousPos;
  }

  // Always prefer placing directly BELOW the clicked entity word
  let top = rect.bottom + 8;

  // Only flip above if there is very little space below AND ample space above
  const spaceBelow = window.innerHeight - rect.bottom - margin;
  if (spaceBelow < 160 && rect.top - margin > 300) {
    top = Math.max(margin, rect.top - 330);
  } else if (top > window.innerHeight - 80) {
    top = Math.max(margin, window.innerHeight - 80);
  }

  const centeredLeft = rect.left + rect.width / 2 - popupWidth / 2;
  const left = Math.max(
    margin,
    Math.min(centeredLeft, window.innerWidth - popupWidth - margin)
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
  /** True when the source API flags the paper as Open Access. */
  isOpenAccess?: boolean;
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
  isOpenAccess = false,
  isDownloadingPdf = false,
  isUploadingToRag = false,
  isAddingToAnalyse = false,
  onDownloadPdf,
  onSendPdfToRag,
  onAddToAnalyse,
  onExtract,
}) => {
  const identifierValue = paperIdentifier?.value || 'paper';

  const [currentSectionIdx, setCurrentSectionIdx] = useState(0);
  const [activeHeading, setActiveHeading] = useState<string | null>(null);
  const [expandedChemical, setExpandedChemical] = useState<string | null>(null);
  void activeHeading;
  const [expandedGroups, setExpandedGroups] = useState<Record<EntityGroupLabel, boolean>>(() => createInitialExpandedGroups());
  const [enabledHighlightGroups, setEnabledHighlightGroups] = useState<Record<EntityGroupLabel, boolean>>(() => createInitialEnabledHighlightGroups());
  const [activeSpeciesPopup, setActiveSpeciesPopup] = useState<SpeciesPopupState | null>(null);
  const [activeChemicalPopup, setActiveChemicalPopup] = useState<ChemicalPopupState | null>(null);
  const titleContainerRef = useRef<HTMLHeadingElement>(null);
  const htmlContainerRef = useRef<HTMLDivElement>(null);
  const isClickScrollingRef = useRef(false);
  const clickScrollTimeoutRef = useRef<number | null>(null);
  const speciesPopupRef = useRef<HTMLDivElement>(null);
  const chemicalPopupRef = useRef<HTMLDivElement>(null);
  const chemicalStructureSvgRef = useRef<SVGSVGElement | null>(null);
  const chemicalStructureRenderIdRef = useRef(0);
  const activeSpeciesAnchorRef = useRef<HTMLElement | null>(null);
  const activeChemicalAnchorRef = useRef<HTMLElement | null>(null);
  const chemicalPopupRequestIdRef = useRef(0);
  const [isRenderingChemicalStructure, setIsRenderingChemicalStructure] = useState(false);
  const [chemicalStructureError, setChemicalStructureError] = useState(false);

  const [tab, setTab] = useState<'entity' | 'graph'>('entity');
  const [hoverTab, setHoverTab] = useState<'entity' | 'graph' | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);
  const [showHL, setShowHL] = useState(true);

  // A map of lowercase entity name/aliases to its uppercase category group
  const entityToGroupMap = useMemo(() => {
    const map = new Map<string, EntityGroupLabel>();
    entities.forEach((entity) => {
      if (entity.text) {
        map.set(entity.text.toLowerCase().replace(/\s+/g, ' '), entity.label as EntityGroupLabel);
      }
      if (entity.canonical) {
        map.set(entity.canonical.toLowerCase().replace(/\s+/g, ' '), entity.label as EntityGroupLabel);
      }
      if (entity.aliases) {
        entity.aliases.forEach(a => map.set(a.toLowerCase().replace(/\s+/g, ' '), entity.label as EntityGroupLabel));
      }
    });
    return map;
  }, [entities]);

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

  const openSpeciesPopup = useCallback((target: HTMLElement, event?: MouseEvent) => {
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
    if (!species) {
      species = {
        primaryName: text,
        metadataScore: 0,
      };
    }

    const fallbackPoint = event ? { x: event.clientX, y: event.clientY } : undefined;
    activeSpeciesAnchorRef.current = target;
    setActiveSpeciesPopup({
      species,
      anchorText: text,
      position: getSpeciesPopupPosition(target, fallbackPoint),
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

  const findMentionsForEntity = useCallback((name: string, aliases?: string[]) => {
    const roots = getInteractiveRoots();
    if (roots.length === 0) return [];

    const targetSet = new Set<string>();
    const addTarget = (val?: string | null) => {
      if (!val) return;
      const clean = stripHtml(val).trim().toLowerCase().replace(/\s+/g, ' ');
      if (clean) {
        targetSet.add(clean);
        if (clean.includes('-')) targetSet.add(clean.replace(/-/g, ' '));
        if (clean.includes(' ')) targetSet.add(clean.replace(/\s+/g, '-'));
      }
    };

    addTarget(name);
    (aliases || []).forEach(addTarget);

    const normalizedName = normalizeLookupText(name);
    const sp = speciesLookup.get(normalizedName);
    if (sp) {
      addTarget(sp.primaryName);
      addTarget(sp.scientificNameVerified);
      addTarget(sp.commonName);
    }
    const ch = chemicalLookup.get(normalizedName);
    if (ch) {
      addTarget(ch.primaryName);
      addTarget(ch.preferredName);
      (ch.synonyms || []).forEach(addTarget);
    }

    const selector = [
      '.ent-species, mark.ner-species',
      '.ent-chemical, mark.ner-chemical',
      '.ent-plant-part, mark.ner-plant-part',
      '.ent-development-stage, mark.ner-development-stage',
      '.ent-extraction-method, mark.ner-extraction-method',
      '.ent-analytical-technique, mark.ner-analytical-technique, .ent-isolation-method, mark.ner-isolation-method',
      '.ent-bioactivity, mark.ner-bioactivity',
      '.ent-disease, mark.ner-disease',
      '.ent-season, mark.ner-season',
      '.ent-location, mark.ner-location'
    ].join(', ');

    const allNodes: HTMLElement[] = [];
    roots.forEach((root) => {
      allNodes.push(...Array.from(root.querySelectorAll<HTMLElement>(selector)));
    });

    return allNodes.filter((node) => {
      const text = stripHtml(node.textContent).trim().toLowerCase().replace(/\s+/g, ' ');
      const dataEnt = (node.getAttribute('data-entity') || '').trim().toLowerCase().replace(/\s+/g, ' ');
      const spKey = (node.dataset.speciesLookupKey || '').trim().toLowerCase().replace(/\s+/g, ' ');
      const chKey = (node.dataset.chemicalLookupKey || '').trim().toLowerCase().replace(/\s+/g, ' ');

      if (
        targetSet.has(text) ||
        (dataEnt && targetSet.has(dataEnt)) ||
        (text.includes('-') && targetSet.has(text.replace(/-/g, ' '))) ||
        (text.includes(' ') && targetSet.has(text.replace(/\s+/g, '-')))
      ) {
        return true;
      }

      const nodeSpecies = speciesLookup.get(normalizeLookupText(node.textContent)) || (spKey ? speciesLookup.get(spKey) : undefined);
      if (nodeSpecies) {
        if (targetSet.has(nodeSpecies.primaryName.toLowerCase()) ||
            (nodeSpecies.scientificNameVerified && targetSet.has(nodeSpecies.scientificNameVerified.toLowerCase())) ||
            (nodeSpecies.commonName && targetSet.has(nodeSpecies.commonName.toLowerCase()))) {
          return true;
        }
      }

      const nodeChem = chemicalLookup.get(normalizeLookupText(node.textContent)) || (chKey ? chemicalLookup.get(chKey) : undefined);
      if (nodeChem) {
        if (targetSet.has(nodeChem.primaryName.toLowerCase()) ||
            (nodeChem.preferredName && targetSet.has(nodeChem.preferredName.toLowerCase()))) {
          return true;
        }
      }

      return false;
    });
  }, [getInteractiveRoots, speciesLookup, chemicalLookup]);

  const pulseEntity = useCallback((name: string, aliases?: string[]) => {
    const key = name.toLowerCase().replace(/\s+/g, ' ');
    const group = entityToGroupMap.get(key);
    const accentVar = group ? ENTITY_GROUP_CONFIG[group].accentVar : '--entity-default';
    
    const els = findMentionsForEntity(name, aliases);
    els.forEach((el) => {
      el.classList.add('entity-pulse');
      el.style.setProperty('--hl-bd', `var(${accentVar})`);
    });
  }, [entityToGroupMap, findMentionsForEntity]);

  const clearPulse = useCallback(() => {
    document.querySelectorAll('.entity-pulse').forEach((el) => {
      el.classList.remove('entity-pulse');
    });
  }, []);

  const openChemicalPopup = useCallback((target: HTMLElement, event?: MouseEvent) => {
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
    if (!chemical) {
      chemical = {
        primaryName: text,
      };
    }

    const fallbackPoint = event ? { x: event.clientX, y: event.clientY } : undefined;
    setIsRenderingChemicalStructure(!!chemical.smiles);
    setChemicalStructureError(false);
    setActiveChemicalPopup({
      chemical,
      anchorText: text,
      position: getChemicalPopupPosition(target, fallbackPoint),
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

    const drawer = new SmilesDrawer.SvgDrawer({ width: 320, height: 180 });

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeChemicalPopup?.chemical.smiles, activeChemicalPopup?.chemical.primaryName]);

  // Group entities by label and calculate true frequency from frontend HTML
  const groupedEntityMap = useMemo(() => {
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

  const sanitizedHtml = useMemo(() => sanitizeHtml(html), [html]);

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

  const graphEntities: Entity[] = useMemo(() => {
    const result: Entity[] = [];
    ENTITY_GROUP_ORDER.forEach((label) => {
      const items = groupedEntityMap[label] || [];
      items.forEach((item) => {
        result.push({
          label,
          text: item.text,
          canonical: item.text,
          count: item.count,
          aliases: item.aliases,
          preferred_name: item.preferred_name,
          smiles: item.smiles,
          inchikey: item.inchikey,
          molecular_formula: item.molecular_formula,
          source_db: item.source_db,
          source_url: item.source_url,
        });
      });
    });
    return result;
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
  }, [closeSpeciesPopup, closeChemicalPopup]);

  useEffect(() => {
    const resetTimer = window.setTimeout(() => {
      resetReaderUiState();
    }, 0);

    return () => window.clearTimeout(resetTimer);
  }, [identifierValue, html, resetReaderUiState]);

  // close the export menu on outside click
  useEffect(() => {
    if (!exportOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setExportOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [exportOpen]);

  const triggerExportCsv = useCallback(() => {
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
    const doiCell = escape(`${paperIdentifier?.type?.toUpperCase() || 'PAPER'}:${identifierValue}`);
    const csv = `DOI,Type,Name,Count,Variants\n${rows.map((row) => `${doiCell},${row}`).join('\n')}`;
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `entities-${identifierValue.replace(/[^a-zA-Z0-9.-]/g, '_')}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [groupedEntityMap, paperIdentifier, identifierValue]);

  const triggerExportGraph = useCallback(() => {
    const nodes = graphEntities.map((e) => ({
      id: `${e.label}-${e.text.toLowerCase()}`,
      label: e.text,
      group: e.label,
    }));
    const edges = graphEntities.map((e) => ({
      from: `paper-${identifierValue}`,
      to: `${e.label}-${e.text.toLowerCase()}`
    }));
    void downloadGraphHtml({
      nodes: [{ id: `paper-${identifierValue}`, label: identifierValue, group: 'PAPER' }, ...nodes],
      edges,
      filename: `graph-${identifierValue.replace(/[^a-zA-Z0-9.-]/g, '_')}.html`,
      title: `Knowledge Graph - ${identifierValue}`,
      subtitle: `Entities linked to ${identifierValue}`,
    });
  }, [graphEntities, identifierValue]);

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
        node.removeAttribute('title');
      });
    });
  }, [
    getInteractiveRoots,
    html,
    speciesLookup,
    activeSpeciesPopup,
    tab,
  ]);

  useEffect(() => {
    const roots = getInteractiveRoots();
    if (roots.length === 0) return;

    const handleSpeciesClick = (event: MouseEvent) => {
      if (!showHL) return;
      const clickedEl = event.target as HTMLElement;
      
      const targetEl = (clickedEl.closest(SPECIES_SELECTOR) || clickedEl.closest('.ent-species')) as HTMLElement | null;
      if (!targetEl) {
        return;
      }
      
      const textContent = targetEl.textContent?.trim();
      if (!textContent) return;
      
      event.preventDefault();
      event.stopPropagation();

      if (activeSpeciesAnchorRef.current === targetEl && activeSpeciesPopup) {
        closeSpeciesPopup();
        return;
      }

      if (activeChemicalPopup) {
        closeChemicalPopup();
      }

      openSpeciesPopup(targetEl, event);
    };

    const handleSpeciesKeyDown = (event: KeyboardEvent) => {
      const target = ((event.target as HTMLElement).closest(SPECIES_SELECTOR) || (event.target as HTMLElement).closest('.ent-species')) as HTMLElement | null;
      if (!target || !roots.some((root) => root.contains(target))) return;

      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
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
  }, [activeSpeciesPopup, activeChemicalPopup, closeSpeciesPopup, closeChemicalPopup, getInteractiveRoots, openSpeciesPopup, speciesLookup, showHL]);

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
        const chemicalFallback = chemicalLookup.get(lookupKey);

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
        node.removeAttribute('title');
      });
    });
  }, [
    getInteractiveRoots,
    html,
    chemicalLookup,
    activeChemicalPopup,
    tab,
  ]);

  // Chemical popup click/key handlers
  useEffect(() => {
    const roots = getInteractiveRoots();
    if (roots.length === 0) return;

    const handleChemicalClick = (event: MouseEvent) => {
      if (!showHL) return;
      const clickedEl = event.target as HTMLElement;
      
      const targetEl = (clickedEl.closest(CHEMICAL_SELECTOR) || clickedEl.closest('.ent-chemical')) as HTMLElement | null;
      if (!targetEl) {
        return;
      }
      
      const textContent = targetEl.textContent?.trim();
      if (!textContent) return;
      
      event.preventDefault();
      event.stopPropagation();

      if (activeChemicalAnchorRef.current === targetEl && activeChemicalPopup) {
        closeChemicalPopup();
        return;
      }

      if (activeSpeciesPopup) {
        closeSpeciesPopup();
      }

      openChemicalPopup(targetEl, event);
    };

    const handleChemicalKeyDown = (event: KeyboardEvent) => {
      const target = ((event.target as HTMLElement).closest(CHEMICAL_SELECTOR) || (event.target as HTMLElement).closest('.ent-chemical')) as HTMLElement | null;
      if (!target || !roots.some((root) => root.contains(target))) return;

      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
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
  }, [activeChemicalPopup, activeSpeciesPopup, closeChemicalPopup, closeSpeciesPopup, getInteractiveRoots, openChemicalPopup, chemicalLookup, showHL]);

  // Reposition active popups on scroll/resize and handle outside click / Escape
  useEffect(() => {
    if (!activeSpeciesPopup && !activeChemicalPopup) return;

    const repositionPopups = () => {
      if (activeSpeciesPopup && activeSpeciesAnchorRef.current) {
        const anchor = activeSpeciesAnchorRef.current;
        if (!anchor.isConnected) {
          closeSpeciesPopup();
          return;
        }
        const rect = anchor.getBoundingClientRect();
        // If scrolled past top header (y < 48) or completely below window, close popup
        if (rect.bottom < 48 || rect.top > window.innerHeight) {
          closeSpeciesPopup();
          return;
        }
        const nextPos = getSpeciesPopupPosition(anchor, undefined, activeSpeciesPopup.position);
        setActiveSpeciesPopup((cur) => {
          if (!cur) return cur;
          if (cur.position.top === nextPos.top && cur.position.left === nextPos.left) return cur;
          return { ...cur, position: nextPos };
        });
      }
      if (activeChemicalPopup && activeChemicalAnchorRef.current) {
        const anchor = activeChemicalAnchorRef.current;
        if (!anchor.isConnected) {
          closeChemicalPopup();
          return;
        }
        const rect = anchor.getBoundingClientRect();
        // If scrolled past top header (y < 48) or completely below window, close popup
        if (rect.bottom < 48 || rect.top > window.innerHeight) {
          closeChemicalPopup();
          return;
        }
        const nextPos = getChemicalPopupPosition(anchor, undefined, activeChemicalPopup.position);
        setActiveChemicalPopup((cur) => {
          if (!cur) return cur;
          if (cur.position.top === nextPos.top && cur.position.left === nextPos.left) return cur;
          return { ...cur, position: nextPos };
        });
      }
    };

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (activeSpeciesPopup) {
        if (!speciesPopupRef.current?.contains(target) && !activeSpeciesAnchorRef.current?.contains(target)) {
          closeSpeciesPopup();
        }
      }
      if (activeChemicalPopup) {
        if (!chemicalPopupRef.current?.contains(target) && !activeChemicalAnchorRef.current?.contains(target)) {
          closeChemicalPopup();
        }
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (activeSpeciesPopup) closeSpeciesPopup();
        if (activeChemicalPopup) closeChemicalPopup();
      }
    };

    const scrollContainer = document.getElementById('main-content-display');
    window.addEventListener('resize', repositionPopups);
    window.addEventListener('scroll', repositionPopups, true);
    scrollContainer?.addEventListener('scroll', repositionPopups, { passive: true });
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleEscape);

    return () => {
      window.removeEventListener('resize', repositionPopups);
      window.removeEventListener('scroll', repositionPopups, true);
      scrollContainer?.removeEventListener('scroll', repositionPopups);
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [activeSpeciesPopup, activeChemicalPopup, closeSpeciesPopup, closeChemicalPopup]);

  // Unified effect to label inline highlight tags — only chemical + species
  // are clickable in the paper (popups); everything else is hover-only.
  useEffect(() => {
    const container = htmlContainerRef.current;
    const titleContainer = titleContainerRef.current;

    if (!isExtracted && (!entities || entities.length === 0)) return;

    const selector = [
      '.ent-species, mark.ner-species',
      '.ent-chemical, mark.ner-chemical',
      '.ent-plant-part, mark.ner-plant-part',
      '.ent-development-stage, mark.ner-development-stage',
      '.ent-extraction-method, mark.ner-extraction-method',
      '.ent-analytical-technique, mark.ner-analytical-technique, .ent-isolation-method, mark.ner-isolation-method',
      '.ent-bioactivity, mark.ner-bioactivity',
      '.ent-disease, mark.ner-disease',
      '.ent-season, mark.ner-season',
      '.ent-location, mark.ner-location'
    ].join(', ');

    const nodes: HTMLElement[] = [];
    if (container) {
      nodes.push(...Array.from(container.querySelectorAll<HTMLElement>(selector)));
    }
    if (titleContainer) {
      nodes.push(...Array.from(titleContainer.querySelectorAll<HTMLElement>(selector)));
    }

    nodes.forEach((node) => {
      node.removeAttribute('title');
      const name = node.textContent?.trim();
      if (!name) return;

      const lowerName = name.toLowerCase().replace(/\s+/g, ' ');
      node.setAttribute('data-entity', lowerName);
      const interactive = node.closest(SPECIES_SELECTOR) || node.closest(CHEMICAL_SELECTOR);
      node.style.cursor = interactive ? 'pointer' : 'default';
    });
  }, [
    html,
    isExtracted,
    entities,
    tab,
  ]);

  // Helper to reliably find any section heading element by ID or text content
  const findHeading = useCallback((targetId: string, targetText?: string): HTMLElement | null => {
    let el = document.getElementById(targetId);
    if (!el && (targetId === 'section-0' || targetId === 'abstract' || targetText?.toLowerCase() === 'abstract')) {
      el = document.getElementById('abstract') || document.getElementById('section-0');
    }
    if (!el && targetText && htmlContainerRef.current) {
      const cleanTarget = targetText.trim().toLowerCase();
      const headings = Array.from(htmlContainerRef.current.querySelectorAll('h1, h2, h3, h4, h5, h6'));
      
      // 1. Exact text match
      el = (headings.find(h => {
        const t = h.textContent?.trim().toLowerCase();
        return t === cleanTarget;
      }) as HTMLElement) || null;

      // 2. Starts with / includes match
      if (!el) {
        el = (headings.find(h => {
          const t = h.textContent?.trim().toLowerCase();
          return t && (t.startsWith(cleanTarget) || cleanTarget.startsWith(t) || t.includes(cleanTarget));
        }) as HTMLElement) || null;
      }

      // 3. Section by id
      if (!el) {
        const sections = Array.from(htmlContainerRef.current.querySelectorAll('section[id]'));
        el = (sections.find(s => s.id.toLowerCase() === targetId.toLowerCase()) as HTMLElement) || null;
      }
    }
    return el;
  }, []);

  // Ensure headings have matching IDs after HTML mounts
  useEffect(() => {
    if (!htmlContainerRef.current || !toc || toc.length === 0) return;
    const container = htmlContainerRef.current;
    const headings = Array.from(container.querySelectorAll('h1, h2, h3, h4, h5, h6, section'));

    toc.forEach((item, idx) => {
      const cleanTarget = item.text.trim().toLowerCase();
      const match = headings.find(h => {
        const t = h.textContent?.trim().toLowerCase();
        if (!t) return false;
        return t === cleanTarget || t.startsWith(cleanTarget) || cleanTarget.startsWith(t) || t.includes(cleanTarget) || cleanTarget.includes(t);
      });
      if (match && !match.id) {
        match.id = item.id || `section-${idx}`;
      }
    });
  }, [html, toc]);

  // Scroll to element by ID or text with container-aware scrolling
  const scrollToId = (id: string, text?: string) => {
    const el = findHeading(id, text);
    if (!el) return;

    isClickScrollingRef.current = true;
    if (clickScrollTimeoutRef.current) window.clearTimeout(clickScrollTimeoutRef.current);
    clickScrollTimeoutRef.current = window.setTimeout(() => {
      isClickScrollingRef.current = false;
    }, 800);

    const scrollContainer = document.getElementById('main-content-display');
    if (scrollContainer) {
      const containerRect = scrollContainer.getBoundingClientRect();
      const elRect = el.getBoundingClientRect();
      const relativeTop = elRect.top - containerRect.top;
      const targetScrollTop = scrollContainer.scrollTop + relativeTop - 24;
      scrollContainer.scrollTo({
        top: Math.max(0, targetScrollTop),
        behavior: 'smooth',
      });
    } else {
      const yOffset = -90;
      const y = el.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: Math.max(0, y), behavior: 'smooth' });
    }
  };

  // Setup real-time scroll listener for active section indicator
  useEffect(() => {
    const scrollContainer = document.getElementById('main-content-display');
    const scrollTarget: EventTarget = scrollContainer || window;

    const handleScroll = () => {
      if (isClickScrollingRef.current) return;
      if (!toc || toc.length === 0 || !htmlContainerRef.current) return;

      const elements: { idx: number; el: HTMLElement }[] = [];
      toc.forEach((item, idx) => {
        const el = findHeading(item.id, item.text);
        if (el) {
          elements.push({ idx, el });
        }
      });

      if (elements.length === 0) return;

      // Sort by vertical position in DOM to guarantee top-to-bottom order
      elements.sort((a, b) => a.el.getBoundingClientRect().top - b.el.getBoundingClientRect().top);

      const containerTop = scrollContainer ? scrollContainer.getBoundingClientRect().top : 0;
      const currentScrollTop = scrollContainer ? scrollContainer.scrollTop : window.scrollY;

      // At the very top: activate section 0
      if (currentScrollTop < 50) {
        setCurrentSectionIdx(0);
        return;
      }

      // At the very bottom: activate last section
      if (scrollContainer) {
        const isAtBottom = scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight < 50;
        if (isAtBottom) {
          setCurrentSectionIdx(elements[elements.length - 1].idx);
          return;
        }
      }

      let activeIdx = elements[0].idx;
      for (const item of elements) {
        const rect = item.el.getBoundingClientRect();
        const relativeTop = rect.top - containerTop;
        if (relativeTop <= 90) {
          activeIdx = item.idx;
        } else {
          break;
        }
      }

      setCurrentSectionIdx(activeIdx);
    };

    scrollTarget.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => {
      scrollTarget.removeEventListener('scroll', handleScroll);
      if (clickScrollTimeoutRef.current) window.clearTimeout(clickScrollTimeoutRef.current);
    };
  }, [html, toc, findHeading]);

  // Citation click handler - REMOVED (references section no longer displayed)


  return (
    <div className="w-full max-w-[1440px] mx-auto px-6 lg:px-14 py-8 paper-enter">

      <div className={`grid grid-cols-1 lg:grid-cols-[190px_minmax(0,1fr)_300px] gap-14 items-start w-full ${showHL ? '' : 'hl-off'}`}>
        
        {/* Left Sidebar: Table of Contents */}
        <aside 
          style={{
            width: 220, flexShrink: 0,
            position: "sticky", top: 232, height: "fit-content",
            paddingRight: 8, marginTop: 39, marginLeft: -40
          }} 
          className="hidden lg:block shrink-0"
        >
          <div style={{
            fontSize: 15, fontWeight: 700, color: "var(--on-surface)",
            marginBottom: 18, paddingLeft: 19,
            fontFamily: "var(--font-google-sans)"
          }}>Sections</div>

          <nav style={{ display: "flex", flexDirection: "column" }}>
            {toc.map((item, idx) => {
              const active = idx === currentSectionIdx;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setCurrentSectionIdx(idx);
                    setActiveHeading(null);
                    scrollToId(item.id, item.text);
                  }}
                  style={{
                    display: "flex", alignItems: "center", gap: 16,
                    width: "100%", textAlign: "left",
                    background: "transparent", border: "none", cursor: "pointer",
                    padding: "9px 0"
                  }}
                  onMouseEnter={(e) => {
                    if (!active) {
                      const span = e.currentTarget.querySelector('span:last-child') as HTMLElement;
                      if (span) span.style.color = "var(--on-surface)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!active) {
                      const span = e.currentTarget.querySelector('span:last-child') as HTMLElement;
                      if (span) span.style.color = "var(--on-surface-variant)";
                    }
                  }}
                >
                  <span style={{
                    width: 3, alignSelf: "stretch", borderRadius: 999,
                    background: active ? "var(--on-surface)" : "transparent",
                    flexShrink: 0
                  }} />
                  <span style={{
                    fontSize: 15,
                    fontWeight: active ? 600 : 400,
                    color: active ? "var(--on-surface)" : "var(--on-surface-variant)",
                    fontFamily: "var(--font-google-sans)",
                    transition: "color .15s"
                  }}>{item.text}</span>
                </button>
              );
            })}
          </nav>
        </aside>

        {/* Main Content Area */}
        <main className="w-full max-w-[990px] min-w-0 flex flex-col bg-background relative">
          
          {/* Metadata Row */}
          <div className="flex items-center gap-2.5 mb-[18px] text-[12.5px] text-on-surface-variant">
            {(() => {
              const srcText = fallbackSource ? fallbackSource.source : mode;
              const lower = (srcText || '').toLowerCase();
              let badgeBg = '#ecfdf5';
              let badgeColor = '#059669';

              if (lower.includes('openalex')) {
                badgeBg = '#f1f5f9';
                badgeColor = '#334155';
              } else if (lower.includes('semantic')) {
                badgeBg = '#fefce8';
                badgeColor = '#a16207';
              } else if (lower.includes('europe')) {
                badgeBg = '#ecfdf5';
                badgeColor = '#059669';
              }

              return (
                <span style={{
                  height: 22, padding: "0 11px", borderRadius: 999,
                  background: badgeBg, color: badgeColor,
                  fontSize: 10.5, fontWeight: 600, letterSpacing: ".04em",
                  display: "inline-flex", alignItems: "center",
                  fontFamily: "var(--font-google-sans)",
                }} className="uppercase">
                  {srcText}
                </span>
              );
            })()}
            {paperIdentifier && (
              <a
                href={paperIdentifier.href}
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-blue-600 hover:text-blue-800 hover:underline transition-colors font-medium"
              >
                {paperIdentifier.value}
              </a>
            )}
            {paperDate && (
              <span className="ml-auto" style={{ fontFamily: "var(--font-google-sans)" }}>
                {paperDate}
              </span>
            )}
          </div>

          {/* Title */}
          <h1
            ref={titleContainerRef}
            data-disabled-entity-groups={disabledHighlightGroupData}
            id="abstract"
            className="text-[26px] lg:text-[30px] font-bold text-on-surface tracking-tight leading-[1.22]"
            style={{ fontFamily: 'Georgia, "Times New Roman", serif', letterSpacing: "-0.01em", marginBottom: 24 }}
          >
            <span dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(title) || 'Untitled Paper' }} />
          </h1>

          {/* Authors & Journal Byline */}
          <div className="flex items-center gap-2.5 mb-[18px] flex-wrap text-[14.5px] text-on-surface-variant" style={{ fontFamily: 'var(--font-google-sans)' }}>
            {paperAuthors.length > 0 && (
              <span className="text-on-surface font-normal">
                {paperAuthors.slice(0, 3).join(', ')}
                {paperAuthors.length > 3 ? ' et al.' : ''}
              </span>
            )}
            {paperAuthors.length > 0 && paperJournal && <span className="text-outline">•</span>}
            {paperJournal && <span className="italic">{paperJournal}</span>}
            
            {/* Status icons — only when the source API actually reports it:
                Open Access comes from Europe PMC/OpenAlex OA flags; Full Text
                only when the reader is really serving the full paper. */}
            {(isOpenAccess || mode === 'full_text') && (
              <span className="ml-auto flex items-center gap-1">
                {isOpenAccess && (
                  <span className="status-ic" title="Open Access" style={{ color: "#E65100" }}>
                    <LockSimpleOpen size={17} weight="regular" />
                  </span>
                )}
                {mode === 'full_text' && (
                  <span className="status-ic" title="Full Text" style={{ color: "#15803D" }}>
                    <Article size={18} weight="regular" color="#1565C0" />
                  </span>
                )}
              </span>
            )}
          </div>

          {/* Action Row */}
          <div className="flex items-center gap-6 mb-7">
            <button 
              onClick={onDownloadPdf} 
              disabled={isDownloadingPdf} 
              style={{ fontFamily: 'var(--font-google-sans)' }}
              className="result-action flex items-center gap-2 bg-transparent text-on-surface-variant hover:text-on-surface text-[13.5px] font-medium transition-colors cursor-pointer"
              title="Download"
            >
              <DownloadSimple size={17} weight="regular" />
              Download
            </button>

            <button 
              onClick={onAddToAnalyse} 
              disabled={isAddingToAnalyse} 
              style={{ fontFamily: 'var(--font-google-sans)' }}
              className="result-action flex items-center gap-2 bg-transparent text-on-surface-variant hover:text-on-surface text-[13.5px] font-medium transition-colors cursor-pointer"
              title="Analyse"
            >
              <ListMagnifyingGlass size={17} weight="regular" />
              Analyse
            </button>

            <button 
              onClick={onSendPdfToRag} 
              disabled={isUploadingToRag} 
              style={{ fontFamily: 'var(--font-google-sans)' }}
              className="result-action flex items-center gap-2 bg-transparent text-on-surface-variant hover:text-on-surface text-[13.5px] font-medium transition-colors cursor-pointer"
              title="Chat"
            >
              <Chats size={17} weight="regular" />
              Chat
            </button>

            {isExtracted && entities && entities.length > 0 && (
              <button 
                type="button"
                onClick={() => {
                  setShowHL(v => {
                    const next = !v;
                    if (!next) {
                      closeSpeciesPopup();
                      closeChemicalPopup();
                    }
                    return next;
                  });
                }} 
                className="status-ic ml-auto text-on-surface-variant hover:text-on-surface transition-colors cursor-pointer border-0 bg-transparent p-0"
                style={{ width: 28, height: 28 }}
                title={showHL ? "Hide highlights" : "Show highlights"}
                aria-label={showHL ? "Hide highlights" : "Show highlights"}
              >
                {showHL ? (
                  <Eye size={17} weight="regular" />
                ) : (
                  <EyeSlash size={17} weight="regular" />
                )}
              </button>
            )}
          </div>

          <div style={{ height: 1, background: "var(--border)", margin: "0 0 32px" }} />

          {/* Section Content */}
          <div
            ref={htmlContainerRef}
            id="section-content-area"
            data-disabled-entity-groups={disabledHighlightGroupData}
            className="article-prose scroll-mt-20"
          >
            <div dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />
          </div>

          {/* Species Popup */}
          {activeSpeciesPopup && typeof document !== 'undefined' && createPortal(
            <div
              ref={speciesPopupRef}
              role="dialog"
              aria-label={`Species metadata for ${activeSpeciesPopup.species.primaryName}`}
              className="fixed z-50 w-[min(20rem,calc(100vw-1.5rem))] max-h-[min(480px,calc(100vh-2rem))] overflow-y-auto rounded-2xl border border-[rgba(22,163,74,0.22)] bg-background p-4 shadow-2xl shadow-on-surface/10 animate-fade-in"
              style={{
                top: `${activeSpeciesPopup.position.top}px`,
                left: `${activeSpeciesPopup.position.left}px`,
                fontFamily: 'var(--font-google-sans)',
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p
                    style={{ fontFamily: 'var(--font-google-sans)', color: 'var(--entity-species, #16A34A)' }}
                    className="text-[12px] font-bold uppercase tracking-[0.14em]"
                  >
                    Species
                  </p>
                  <h3
                    style={{ fontFamily: 'var(--font-google-sans)' }}
                    className="mt-1.5 text-[19px] font-bold text-on-surface italic leading-snug"
                  >
                    {activeSpeciesPopup.species.scientificNameVerified || activeSpeciesPopup.species.primaryName}
                  </h3>
                  {activeSpeciesPopup.species.commonName && normalizeLookupText(activeSpeciesPopup.species.commonName) !== normalizeLookupText(activeSpeciesPopup.species.primaryName) && (
                    <p
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="mt-1 text-[14.5px] font-medium text-on-surface-variant normal-case"
                    >
                      {activeSpeciesPopup.species.commonName}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={closeSpeciesPopup}
                  aria-label="Close"
                  className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-100 text-slate-500 transition-colors hover:bg-slate-200 hover:text-slate-800 cursor-pointer shrink-0"
                >
                  <X size={15} weight="bold" />
                </button>
              </div>

              <div className="mt-4 space-y-2.5 border-t border-border pt-3">
                {activeSpeciesPopup.species.taxonId && (
                  <div className="flex items-center gap-3">
                    <span
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="w-[72px] shrink-0 text-[12px] font-bold uppercase tracking-[0.14em] text-on-surface-variant"
                    >
                      Taxon
                    </span>
                    <span
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="min-w-0 break-words font-medium text-[13px] text-on-surface"
                    >
                      {activeSpeciesPopup.species.taxonId}
                    </span>
                  </div>
                )}
                {(activeSpeciesPopup.species.sourceDb || activeSpeciesPopup.species.sourceUrl) && (
                  <div className="flex items-center gap-3">
                    <span
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="w-[72px] shrink-0 text-[12px] font-bold uppercase tracking-[0.14em] text-on-surface-variant"
                    >
                      Source
                    </span>
                    {activeSpeciesPopup.species.sourceUrl ? (
                      <a
                        href={activeSpeciesPopup.species.sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ fontFamily: 'var(--font-google-sans)' }}
                        className="min-w-0 break-words font-medium text-[13px] text-blue-600 hover:text-blue-800 hover:underline"
                      >
                        {activeSpeciesPopup.species.sourceDb || 'View'}
                      </a>
                    ) : (
                      <span
                        style={{ fontFamily: 'var(--font-google-sans)' }}
                        className="min-w-0 break-words font-medium text-[13px] text-on-surface"
                      >
                        {activeSpeciesPopup.species.sourceDb}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>,
            document.body
          )}

          {/* Chemical Popup */}
          {activeChemicalPopup && typeof document !== 'undefined' && createPortal(
            <div
              ref={chemicalPopupRef}
              role="dialog"
              aria-label={`Chemical metadata for ${activeChemicalPopup.chemical.primaryName}`}
              className="fixed z-50 w-[min(20rem,calc(100vw-1.5rem))] max-h-[min(500px,calc(100vh-2rem))] overflow-y-auto rounded-2xl border border-[rgba(37,99,235,0.22)] bg-background p-4 shadow-2xl shadow-on-surface/10 animate-fade-in"
              style={{
                top: `${activeChemicalPopup.position.top}px`,
                left: `${activeChemicalPopup.position.left}px`,
                fontFamily: 'var(--font-google-sans)',
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p
                    style={{ fontFamily: 'var(--font-google-sans)', color: 'var(--entity-chemical, #2563EB)' }}
                    className="text-[12px] font-bold uppercase tracking-[0.14em]"
                  >
                    Chemical
                  </p>
                  <h3
                    style={{ fontFamily: 'var(--font-google-sans)' }}
                    className="mt-1.5 text-[19px] font-bold text-on-surface leading-snug"
                  >
                    {activeChemicalPopup.chemical.primaryName}
                  </h3>
                  {activeChemicalPopup.chemical.preferredName && normalizeLookupText(activeChemicalPopup.chemical.preferredName) !== normalizeLookupText(activeChemicalPopup.chemical.primaryName) && (
                    <p
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="mt-1 text-[14.5px] font-medium text-on-surface-variant normal-case"
                    >
                      {activeChemicalPopup.chemical.preferredName}
                    </p>
                  )}
                  {activeChemicalPopup.chemical.synonyms && activeChemicalPopup.chemical.synonyms.length > 0 && (
                    <div
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="mt-1.5 text-[12px] font-medium text-on-surface-muted normal-case"
                    >
                      Also: {isExpandedChemical ? (
                        <>
                          {activeChemicalPopup.chemical.synonyms.join(', ')}
                          <button
                            type="button"
                            onClick={() => toggleExpandedChemical(activeChemicalPopup.chemical.primaryName)}
                            className="ml-1 font-semibold text-blue-600 hover:underline cursor-pointer"
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
                              className="ml-1 font-semibold text-blue-600 hover:underline cursor-pointer"
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
                  aria-label="Close"
                  className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-100 text-slate-500 transition-colors hover:bg-slate-200 hover:text-slate-800 cursor-pointer shrink-0"
                >
                  <X size={15} weight="bold" />
                </button>
              </div>

              {/* Metadata Rows — arranged immediately under header with divider, identical to species popup */}
              <div className="mt-4 space-y-2.5 border-t border-border pt-3">
                {activeChemicalPopup.chemical.molecularFormula && (
                  <div className="flex items-center gap-3">
                    <span
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="w-[72px] shrink-0 text-[12px] font-bold uppercase tracking-[0.14em] text-on-surface-variant"
                    >
                      Formula
                    </span>
                    <span
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="min-w-0 break-words font-medium text-[13px] text-on-surface"
                    >
                      {activeChemicalPopup.chemical.molecularFormula}
                    </span>
                  </div>
                )}
                {activeChemicalPopup.chemical.inchikey && (
                  <div className="flex items-start gap-3">
                    <span
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="w-[72px] shrink-0 text-[12px] font-bold uppercase tracking-[0.14em] text-on-surface-variant"
                    >
                      InChIKey
                    </span>
                    <span
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="min-w-0 break-words font-medium text-[12px] text-on-surface break-all"
                    >
                      {activeChemicalPopup.chemical.inchikey}
                    </span>
                  </div>
                )}
                {(activeChemicalPopup.chemical.sourceDb || activeChemicalPopup.chemical.sourceUrl) && (
                  <div className="flex items-center gap-3">
                    <span
                      style={{ fontFamily: 'var(--font-google-sans)' }}
                      className="w-[72px] shrink-0 text-[12px] font-bold uppercase tracking-[0.14em] text-on-surface-variant"
                    >
                      Source
                    </span>
                    {activeChemicalPopup.chemical.sourceUrl ? (
                      <a
                        href={activeChemicalPopup.chemical.sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ fontFamily: 'var(--font-google-sans)' }}
                        className="min-w-0 break-words font-medium text-[13px] text-blue-600 hover:text-blue-800 hover:underline"
                      >
                        {activeChemicalPopup.chemical.sourceDb || 'View'}
                      </a>
                    ) : (
                      <span
                        style={{ fontFamily: 'var(--font-google-sans)' }}
                        className="min-w-0 break-words font-medium text-[13px] text-on-surface"
                      >
                        {activeChemicalPopup.chemical.sourceDb}
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Molecular Structure — properly boxed */}
              {activeChemicalPopup.chemical.smiles && (
                <div className="mt-3.5 rounded-xl border border-slate-200/90 bg-slate-50/70 p-2.5">
                  <div className="relative flex h-[130px] w-full items-center justify-center overflow-hidden">
                    {isRenderingChemicalStructure && (
                      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 rounded-lg bg-background/80 backdrop-blur-[1px]">
                        <SpinnerGap size={20} className="animate-spin text-slate-800" />
                        <p style={{ fontFamily: 'var(--font-google-sans)' }} className="text-xs font-medium text-on-surface-variant">Rendering structure…</p>
                      </div>
                    )}
                    {!chemicalStructureError ? (
                      <svg
                        ref={chemicalStructureSvgRef}
                        className="h-full w-full max-h-[120px] object-contain"
                        viewBox="0 0 320 180"
                        aria-label={`${activeChemicalPopup.chemical.primaryName} molecular structure`}
                      />
                    ) : (
                      <div style={{ fontFamily: 'var(--font-google-sans)' }} className="text-xs text-on-surface-muted">Structure unavailable</div>
                    )}
                  </div>
                </div>
              )}
            </div>,
            document.body
          )}

          {isFetchingFallback && (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <SpinnerGap size={24} className="animate-spin text-slate-900 mx-auto mb-3" />
                <p className="text-xs text-on-surface-muted">Fetching abstract from alternative sources...</p>
              </div>
            </div>
          )}
        </main>

        {/* Right Sidebar: Entity Groups */}
        <aside className="w-full lg:w-[330px] sticky top-[88px] h-fit flex flex-col gap-3.5 shrink-0 z-30" style={{ fontFamily: "var(--font-google-sans)" }}>
          {!isExtracted ? (
            <div style={{
              display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center",
              gap: 16, padding: "32px 24px",
              borderRadius: 16,
              background: "#FFFFFF"
            }}>
              <span style={{
                width: 44, height: 44, borderRadius: 12,
                display: "grid", placeItems: "center",
                background: "#e0f7fa",
                color: "var(--on-surface)"
              }} className={isExtracting ? "animate-pulse" : ""}>
                <Sparkle size={22} weight="fill" />
              </span>
              <button 
                onClick={() => {
                  if (onExtract && !isExtracting) onExtract();
                }}
                disabled={isExtracting}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 9,
                  height: 44, padding: "0 24px", width: "100%",
                  borderRadius: 999,
                  background: "#e0f7fa", color: "var(--on-surface)",
                  border: "none", cursor: isExtracting ? "wait" : "pointer",
                  fontSize: 15, fontWeight: 600, letterSpacing: ".01em",
                  fontFamily: "var(--font-google-sans)",
                  transition: "opacity .15s",
                  opacity: isExtracting ? 0.7 : 1
                }}
                onMouseEnter={(e) => { if (!isExtracting) e.currentTarget.style.opacity = "0.9"; }}
                onMouseLeave={(e) => { if (!isExtracting) e.currentTarget.style.opacity = "1"; }}
              >
                {isExtracting ? (
                  <>
                    <SpinnerGap size={16} className="animate-spin" />
                    Extracting...
                  </>
                ) : (
                  <>
                    Extract Entities
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="flex flex-col flex-1 mt-2">
              {/* Centered view switcher pill header */}
              <div style={{ position: "relative", display: "flex", justifyContent: "center", minHeight: 44, marginBottom: 14 }}>
                <div style={{
                  display: "inline-flex", gap: 3,
                  padding: 4, borderRadius: 999,
                  border: "none", background: "var(--surface-c)"
                }}>
                  {(
                    [
                      { id: 'entity', icon: ListBullets, label: 'Entity Index' },
                      { id: 'graph', icon: Graph, label: 'Graph View' }
                    ] as const
                  ).map(({ id, icon: Icon, label }) => {
                    const isActive = tab === id;
                    const expanded = hoverTab === id;
                    return (
                      <button key={id}
                        data-tab={id}
                        onClick={() => setTab(id)}
                        onMouseEnter={() => setHoverTab(id)}
                        onMouseLeave={() => setHoverTab(null)}
                        style={{
                          display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 7,
                          height: 36, borderRadius: 999,
                          width: expanded ? "auto" : 40,
                          padding: expanded ? "0 14px" : 0,
                          background: isActive ? "var(--background, #FFFFFF)" : "transparent",
                          boxShadow: isActive ? "0 1px 3px rgba(0,0,0,.08)" : "none",
                          border: "none", cursor: "pointer",
                          fontSize: 13.5, fontWeight: isActive ? 600 : 500,
                          fontFamily: "var(--font-google-sans)",
                          color: isActive ? "var(--on-surface)" : "var(--on-surface-variant)",
                          whiteSpace: "nowrap", overflow: "hidden",
                          transition: "width .22s ease, padding .22s ease, background .15s, color .15s"
                        }}>
                          {expanded ? (
                            <span className="inline-flex items-center gap-2" style={{ fontFamily: "var(--font-google-sans)" }}>
                              <Icon size={18} weight={isActive ? "bold" : "regular"} />
                              <span style={{ fontSize: 13.5, fontWeight: isActive ? 600 : 500 }}>{label}</span>
                            </span>
                          ) : (
                            <Icon size={19} weight={isActive ? "bold" : "regular"} />
                          )}
                        </button>
                    );
                  })}
                </div>

                {/* Export options pinned right */}
                <div
                  ref={exportRef}
                  style={{ position: "absolute", right: 0, top: 0 }}
                >
                  <button 
                    onClick={() => setExportOpen((o) => !o)} 
                    title="Export" 
                    aria-label="Export"
                    style={{
                      width: 40, height: 40, borderRadius: 999,
                      background: exportOpen ? "var(--surface-c)" : "transparent",
                      color: "var(--on-surface-variant)",
                      border: "none", cursor: "pointer",
                      display: "grid", placeItems: "center",
                      transition: "background .15s"
                    }}
                  >
                    <DotsThreeVertical size={20} weight="bold" />
                  </button>
                  {exportOpen && (
                    <div style={{
                      position: "absolute", top: "calc(100% + 4px)", right: 0,
                      width: "max-content",
                      background: "#FFFFFF",
                      border: "1px solid var(--border)", borderRadius: 12,
                      boxShadow: "0 2px 6px 2px rgba(0, 0, 0, 0.08)",
                      padding: 5, zIndex: 100,
                      fontFamily: "var(--font-google-sans)",
                      animation: "fadeUp .16s ease"
                    }}>
                      <button 
                        className="export-opt" 
                        style={{ fontFamily: "var(--font-google-sans)" }}
                        onClick={() => {
                          setExportOpen(false);
                          triggerExportCsv();
                        }}
                      >
                        <span style={{ fontFamily: "var(--font-google-sans)" }}>Export CSV</span>
                      </button>
                      <button 
                        className="export-opt" 
                        style={{ fontFamily: "var(--font-google-sans)" }}
                        onClick={() => {
                          setExportOpen(false);
                          triggerExportGraph();
                        }}
                      >
                        <span style={{ fontFamily: "var(--font-google-sans)" }}>Export Graph</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Accordion + Graph share one anchored scroll region, so switching
                  tabs or expanding groups never moves surrounding content */}
              <div className="overflow-y-auto scrollbar-hide min-h-[380px] h-[min(560px,calc(100vh-260px))]">
              {tab === 'entity' ? (
                <div style={{ display: "flex", flexDirection: "column" }}>
                  {visibleGroupedEntities.map((group) => {
                    const isExpanded = expandedGroups[group.label];
                    const accentColor = `var(${group.accentVar})`;
                    const empty = group.termCount === 0;

                    return (
                      <div 
                        key={group.label} 
                        style={{ borderBottom: "1px solid var(--border)" }}
                      >
                        {/* Accordion Row (.ent-cat) */}
                        <button 
                          type="button"
                          className="ent-cat group"
                          onClick={empty ? undefined : () => {
                            toggleEntityGroup(group.label);
                          }}
                          style={{
                            display: "flex", alignItems: "center", gap: 12,
                            width: "100%", padding: "13px 4px",
                            background: "transparent", border: "none",
                            cursor: empty ? "default" : "pointer",
                            opacity: empty ? 0.5 : 1,
                            fontFamily: "var(--font-google-sans)"
                          }}
                        >
                          {/* Left category accent dot bar */}
                          <span style={{
                            width: 4, height: 20, borderRadius: 999,
                            background: accentColor, flexShrink: 0,
                            opacity: empty ? 0.3 : 1
                          }} />
                          
                          <span style={{
                            flex: 1, textAlign: "left", fontSize: 14.5, fontWeight: 600,
                            color: isExpanded ? accentColor : "var(--on-surface)",
                            transition: "color .15s",
                            textTransform: "capitalize",
                            fontFamily: "var(--font-google-sans)"
                          }}>
                            {group.label.toLowerCase()}
                          </span>

                          {/* Hover caret transitions slot */}
                          <span style={{ flexShrink: 0, fontFamily: "var(--font-google-sans)" }}>
                            {empty ? (
                              <span style={{ fontSize: 14, color: "var(--on-surface-variant)", fontFamily: "var(--font-google-sans)" }}>–</span>
                            ) : isExpanded ? (
                              <span className="relative flex items-center justify-center min-w-[24px] h-[20px] text-on-surface-variant">
                                <CaretUp size={16} weight="bold" />
                              </span>
                            ) : (
                              <span className="relative flex items-center justify-center min-w-[24px] h-[20px]">
                                {/* Entity Count: visible without cursor, fades out on hover */}
                                <span className="transition-all duration-200 text-[13px] font-semibold group-hover:opacity-0 group-hover:scale-75" style={{ color: "var(--on-surface)", fontFamily: "var(--font-google-sans)" }}>
                                  {group.termCount}
                                </span>
                                
                                {/* Caret: animates into view on hover */}
                                <span className="absolute inset-0 m-auto flex items-center justify-center transition-all duration-200 text-on-surface-variant opacity-0 scale-75 group-hover:opacity-100 group-hover:scale-100">
                                  <CaretDown size={16} weight="bold" />
                                </span>
                              </span>
                            )}
                          </span>
                        </button>

                        {/* Values expanded container — animated open/close */}
                        {!empty && (
                          <div
                            className="grid transition-[grid-template-rows,opacity] duration-300 ease-out"
                            style={{
                              gridTemplateRows: isExpanded ? '1fr' : '0fr',
                              opacity: isExpanded ? 1 : 0,
                            }}
                          >
                          <div className="min-h-0 overflow-hidden">
                          <div style={{ padding: "0 4px 12px 16px", display: "flex", flexDirection: "column", gap: 2 }}>
                            {group.visibleItems.map((ent, eIdx) => {
                              const name = ent.text;

                              return (
                                <div 
                                  key={eIdx}
                                  className="ent-row group/item"
                                  style={{
                                    display: "flex", alignItems: "center", gap: 10,
                                    padding: "7px 10px", borderRadius: 7, cursor: "default",
                                    transition: "background .12s ease",
                                    background: "transparent",
                                    fontFamily: "var(--font-google-sans)"
                                  }}
                                  onMouseEnter={(e) => {
                                    e.currentTarget.style.background = "var(--surface-low)";
                                    pulseEntity(name, ent.aliases);
                                  }}
                                  onMouseLeave={(e) => {
                                    e.currentTarget.style.background = "transparent";
                                    clearPulse();
                                  }}
                                >
                                  {/* Item dot */}
                                  <span style={{ width: 7.5, height: 7.5, borderRadius: 999, background: accentColor, flexShrink: 0 }} />
                                  
                                  {/* Item name (italicized for Species label) */}
                                  <span style={{
                                    flex: 1, fontSize: 14,
                                    lineHeight: "1.35",
                                    color: "var(--on-surface)",
                                    fontStyle: group.label === "SPECIES" ? "italic" : "normal",
                                    fontFamily: "var(--font-google-sans)",
                                    fontWeight: 400,
                                  }}>
                                    {name}
                                  </span>

                                  {/* Total count in default black color */}
                                  <span style={{ fontSize: 12.5, color: "var(--on-surface)", fontFamily: "var(--font-google-sans)", fontWeight: 500 }}>
                                    {ent.count}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                          </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="flex flex-col">
                  <KnowledgeGraph 
                    entities={graphEntities} 
                    paperIdentifier={paperIdentifier}
                    entityConfig={ENTITY_GROUP_CONFIG}
                  />
                </div>
              )}
              </div>
            </div>
          )}

          {extractionError && (
            <div className="p-4 bg-red-50 border border-red-100 rounded-xl mt-4">
              <p className="text-[10px] text-red-600 font-bold uppercase tracking-wider mb-1">Extraction Error</p>
              <p className="text-[11px] text-red-500 leading-relaxed font-medium">{extractionError}</p>
            </div>
          )}
        </aside>
      </div>

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
