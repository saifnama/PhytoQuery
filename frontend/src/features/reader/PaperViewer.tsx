import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { sanitizeHtml, formatTextWithFormatting } from '../../utils/sanitize';
import { Sparkle, ListBullets, Graph, DotsThreeVertical, DownloadSimple, ListMagnifyingGlass, Chats, Eye, EyeSlash, LockSimpleOpen, Article } from '@phosphor-icons/react';
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

  const [tab, setTab] = useState<'entity' | 'graph'>('entity');
  const [hoverTab, setHoverTab] = useState<'entity' | 'graph' | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);
  const [showHL, setShowHL] = useState(true);

  const [nav, setNav] = useState<{ name: string; idx: number; total: number } | null>(null);
  const mentionsRef = useRef<HTMLElement[]>([]);

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

  const activateEntity = useCallback((name: string) => {
    const key = name.toLowerCase().replace(/\s+/g, ' ');
    const group = entityToGroupMap.get(key);
    if (group) {
      setExpandedGroups(prev => ({ ...prev, [group]: true }));
    }

    const els = Array.from(document.querySelectorAll(`[data-entity="${key}"]`)) as HTMLElement[];
    mentionsRef.current = els;
    if (els.length > 0) {
      setNav({ name, idx: 0, total: els.length });
      
      const rect = els[0].getBoundingClientRect();
      const scrollContainer = document.getElementById('main-content-display');
      if (scrollContainer) {
        scrollContainer.scrollTo({
          top: rect.top + scrollContainer.scrollTop - 160,
          behavior: 'smooth'
        });
      } else {
        window.scrollTo({
          top: rect.top + window.scrollY - 160,
          behavior: 'smooth'
        });
      }
      
      els.forEach(el => el.classList.remove('entity-flash'));
      void els[0].offsetWidth; // force reflow
      els[0].classList.add('entity-flash');
      setTimeout(() => els[0].classList.remove('entity-flash'), 1600);
    }
  }, [entityToGroupMap]);

  const gotoMention = useCallback((i: number, total: number) => {
    const els = mentionsRef.current;
    if (!els.length) return;
    const idx = (i + total) % total;
    const el = els[idx];
    
    const scrollContainer = document.getElementById('main-content-display');
    if (scrollContainer) {
      scrollContainer.scrollTo({
        top: el.getBoundingClientRect().top + scrollContainer.scrollTop - 160,
        behavior: 'smooth'
      });
    } else {
      window.scrollTo({
        top: el.getBoundingClientRect().top + window.scrollY - 160,
        behavior: 'smooth'
      });
    }
    
    els.forEach(e => e.classList.remove('entity-flash'));
    void el.offsetWidth; // force reflow
    el.classList.add('entity-flash');
    
    setNav(n => n ? { ...n, idx } : n);
  }, []);

  const pulseEntity = useCallback((name: string) => {
    const key = name.toLowerCase().replace(/\s+/g, ' ');
    const group = entityToGroupMap.get(key);
    const accentVar = group ? ENTITY_GROUP_CONFIG[group].accentVar : '--entity-default';
    
    document.querySelectorAll(`[data-entity="${key}"]`).forEach(el => {
      el.classList.add('entity-pulse');
      (el as HTMLElement).style.setProperty('--hl-bd', `var(${accentVar})`);
    });
  }, [entityToGroupMap]);

  const clearPulse = useCallback(() => {
    document.querySelectorAll('.entity-pulse').forEach(el => {
      el.classList.remove('entity-pulse');
    });
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
    const csv = `# ${paperIdentifier?.type?.toUpperCase() || 'PAPER'}: ${identifierValue}\nType,Name,Count,Variants\n${rows.join('\n')}`;
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `entities-${identifierValue.replace(/[^a-zA-Z0-9.-]/g, '_')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [groupedEntityMap, paperIdentifier, identifierValue]);

  const triggerExportGraph = useCallback(() => {
    const nodes = entities.map((e) => ({
      id: `${e.label}-${e.text.toLowerCase()}`,
      label: e.text,
      type: e.label,
      count: (e as Entity & { count?: number }).count || 1
    }));
    const edges = entities.map((e) => ({
      from: `paper-${identifierValue}`,
      to: `${e.label}-${e.text.toLowerCase()}`
    }));
    const graphData = {
      paper: {
        type: paperIdentifier?.type,
        value: identifierValue,
        title: title
      },
      nodes,
      edges
    };
    const blob = new Blob([JSON.stringify(graphData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `graph-${identifierValue.replace(/[^a-zA-Z0-9.-]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [entities, paperIdentifier, identifierValue, title]);

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
  }, [
    getInteractiveRoots,
    html,
    speciesLookup,
    activeSpeciesPopup,
    tab,
    showHL,
    nav,
    expandedGroups,
    activeChemicalPopup
  ]);

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
  }, [
    getInteractiveRoots,
    html,
    chemicalLookup,
    activeChemicalPopup,
    tab,
    showHL,
    nav,
    expandedGroups,
    activeSpeciesPopup
  ]);

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

  // Unified effect to dynamically label and bind click events to all inline highlight tags
  useEffect(() => {
    const container = htmlContainerRef.current;
    const titleContainer = titleContainerRef.current;

    // Debug logging
    (window as any).__phytoquery_debug = {
      mounted: true,
      containerExists: !!container,
      titleContainerExists: !!titleContainer,
      isExtracted,
      htmlLength: html?.length || 0,
      nodesFound: 0
    };

    if (!isExtracted) return;

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

    (window as any).__phytoquery_debug.nodesFound = nodes.length;
    (window as any).__phytoquery_debug.matchedTags = nodes.slice(0, 5).map(n => ({
      tag: n.tagName,
      text: n.textContent?.trim(),
      classes: n.className,
      hasDataEntity: n.hasAttribute('data-entity'),
      dataEntityVal: n.getAttribute('data-entity')
    }));

    const clickHandlers: { node: HTMLElement; handler: (e: MouseEvent) => void }[] = [];

    nodes.forEach((node) => {
      const name = node.textContent?.trim();
      if (!name) return;
      
      const lowerName = name.toLowerCase().replace(/\s+/g, ' ');
      node.setAttribute('data-entity', lowerName);
      node.style.cursor = 'pointer';

      const handler = () => {
        activateEntity(name);
      };
      
      node.addEventListener('click', handler);
      clickHandlers.push({ node, handler });
    });

    if (nodes.length > 0) {
      (window as any).__phytoquery_debug.node1_outerHTML_after_setAttribute = nodes[0].outerHTML;
    }

    return () => {
      clickHandlers.forEach(({ node, handler }) => {
        if (node.isConnected) {
          node.removeEventListener('click', handler);
        }
      });
    };
  }, [
    html,
    isExtracted,
    entities,
    activateEntity,
    tab,
    showHL,
    nav,
    expandedGroups,
    activeSpeciesPopup,
    activeChemicalPopup
  ]);

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

  const miniNavStyle: React.CSSProperties = {
    width: 20,
    height: 20,
    borderRadius: 6,
    display: "grid",
    placeItems: "center",
    background: "var(--surface-c)",
    color: "var(--on-surface)",
    border: "none",
    cursor: "pointer",
    fontSize: 8,
    fontWeight: "bold"
  };

  return (
    <div className="w-full max-w-[1440px] mx-auto px-6 lg:px-14 py-8 paper-enter">

      <div className={`grid grid-cols-1 lg:grid-cols-[190px_minmax(0,1fr)_300px] gap-14 items-start w-full ${showHL ? '' : 'hl-off'}`}>
        
        {/* Left Sidebar: Table of Contents */}
        <aside 
          style={{
            width: 220, flexShrink: 0,
            position: "sticky", top: 232, height: "fit-content",
            paddingRight: 8, marginTop: 39, marginLeft: -12
          }} 
          className="hidden lg:block shrink-0"
        >
          <div style={{
            fontSize: 15, fontWeight: 500, color: "var(--on-surface)",
            marginBottom: 18, paddingLeft: 19
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
                    scrollToId(item.id);
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
                    fontSize: 15.5,
                    fontWeight: active ? 700 : 400,
                    color: active ? "var(--on-surface)" : "var(--on-surface-variant)",
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
            <span style={{
              height: 22, padding: "0 11px", borderRadius: 999,
              background: "#ECFDF3", color: "#16794C",
              fontSize: 10.5, fontWeight: 600, letterSpacing: ".04em",
              display: "inline-flex", alignItems: "center"
            }} className="uppercase">
              {fallbackSource ? fallbackSource.source : mode}
            </span>
            {paperIdentifier && (
              <a
                href={paperIdentifier.href}
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-on-surface-variant hover:text-primary transition-colors uppercase"
              >
                {paperIdentifier.type.toUpperCase()}: {paperIdentifier.value}
              </a>
            )}
            {paperDate && (
              <span className="ml-auto">
                {paperDate}
              </span>
            )}
          </div>

          {/* Title */}
          <h1
            ref={titleContainerRef}
            data-disabled-entity-groups={disabledHighlightGroupData}
            id="abstract"
            className="text-3xl lg:text-[38px] font-bold text-on-surface tracking-tight leading-[1.18] mb-5"
            style={{ fontFamily: 'Georgia, "Times New Roman", serif', letterSpacing: "-0.01em" }}
          >
            <span dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(title) || 'Untitled Paper' }} />
          </h1>

          {/* Authors & Journal Byline */}
          <div className="flex items-center gap-2.5 mb-[18px] flex-wrap text-[14.5px] text-on-surface-variant">
            {paperAuthors.length > 0 && (
              <span className="text-on-surface">
                <strong>{paperAuthors.slice(0, 3).join(', ')}</strong>
                {paperAuthors.length > 3 ? ' et al.' : ''}
              </span>
            )}
            {paperAuthors.length > 0 && paperJournal && <span className="text-outline">•</span>}
            {paperJournal && <span className="italic">{paperJournal}</span>}
            
            {/* Quiet Status Icons on the right */}
            <span className="ml-auto flex items-center gap-1">
              <span className="status-ic" title="Open Access" style={{ color: "#E65100" }}>
                <LockSimpleOpen size={17} weight="regular" />
              </span>
              <span className="status-ic" title="Full text available" style={{ color: "#15803D" }}>
                <Article size={18} weight="regular" color="#1565C0" />
              </span>
            </span>
          </div>

          {/* Action Row */}
          <div className="flex items-center gap-6 mb-7">
            <button 
              onClick={onDownloadPdf} 
              disabled={isDownloadingPdf} 
              className="result-action flex items-center gap-2 bg-transparent text-on-surface-variant hover:text-on-surface text-[13.5px] font-medium transition-colors"
              title="Download"
            >
              <DownloadSimple size={17} weight="regular" />
              Download
            </button>

            <button 
              onClick={onAddToAnalyse} 
              disabled={isAddingToAnalyse} 
              className="result-action flex items-center gap-2 bg-transparent text-on-surface-variant hover:text-on-surface text-[13.5px] font-medium transition-colors"
              title="Analyse"
            >
              <ListMagnifyingGlass size={17} weight="regular" />
              Analyse
            </button>

            <button 
              onClick={onSendPdfToRag} 
              disabled={isUploadingToRag} 
              className="result-action flex items-center gap-2 bg-transparent text-on-surface-variant hover:text-on-surface text-[13.5px] font-medium transition-colors"
              title="Chat"
            >
              <Chats size={17} weight="regular" />
              Chat
            </button>

            <button 
              onClick={() => setShowHL(v => !v)} 
              className="result-action flex items-center gap-2 bg-transparent text-on-surface-variant hover:text-on-surface text-[13.5px] font-medium transition-colors ml-auto p-1"
              title={showHL ? "Hide highlights" : "Show highlights"}
            >
              {showHL ? (
                <Eye size={18} weight="regular" />
              ) : (
                <EyeSlash size={18} weight="regular" />
              )}
            </button>
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
          {activeSpeciesPopup && (
            <div
              ref={speciesPopupRef}
              role="dialog"
              aria-label={`Species metadata for ${activeSpeciesPopup.species.primaryName}`}
              className="fixed z-40 w-[min(20rem,calc(100vw-1.5rem))] rounded-2xl border border-teal-100 bg-background p-4 shadow-2xl shadow-on-surface/10 animate-fade-in"
              style={{
                top: `${activeSpeciesPopup.position.top}px`,
                left: `${activeSpeciesPopup.position.left}px`,
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-teal-600">
                    Species
                  </p>
                  <h3 className="mt-2 text-sm font-semibold text-on-surface italic leading-snug">
                    {activeSpeciesPopup.species.scientificNameVerified || activeSpeciesPopup.species.primaryName}
                  </h3>
                  {activeSpeciesPopup.species.commonName && normalizeLookupText(activeSpeciesPopup.species.commonName) !== normalizeLookupText(activeSpeciesPopup.species.primaryName) && (
                    <p className="mt-1 text-[11px] font-medium text-on-surface-variant normal-case">
                      {activeSpeciesPopup.species.commonName}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={closeSpeciesPopup}
                  className="rounded-lg border border-border px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-on-surface-muted transition-colors hover:border-outline hover:text-on-surface-variant cursor-pointer"
                >
                  Close
                </button>
              </div>

              <div className="mt-4 space-y-2 border-t border-border pt-3 text-[10px] text-on-surface-variant">
                {activeSpeciesPopup.species.taxonId && (
                  <div className="flex items-start gap-2">
                    <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-on-surface-muted">Taxon</span>
                    <span className="min-w-0 break-words font-medium">{activeSpeciesPopup.species.taxonId}</span>
                  </div>
                )}
                {(activeSpeciesPopup.species.sourceDb || activeSpeciesPopup.species.sourceUrl) && (
                  <div className="flex items-start gap-2">
                    <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-on-surface-muted">Source</span>
                    {activeSpeciesPopup.species.sourceUrl ? (
                      <a
                        href={activeSpeciesPopup.species.sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="min-w-0 break-words font-medium text-primary hover:underline"
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

          {/* Chemical Popup */}
          {activeChemicalPopup && (
            <div
              ref={chemicalPopupRef}
              role="dialog"
              aria-label={`Chemical metadata for ${activeChemicalPopup.chemical.primaryName}`}
              className="fixed z-40 w-[min(20rem,calc(100vw-1.5rem))] rounded-2xl border border-primary/20 bg-background p-4 shadow-2xl shadow-on-surface/10 animate-fade-in"
              style={{
                top: `${activeChemicalPopup.position.top}px`,
                left: `${activeChemicalPopup.position.left}px`,
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-primary">
                    Chemical
                  </p>
                  <h3 className="mt-2 text-sm font-semibold text-on-surface leading-snug">
                    {activeChemicalPopup.chemical.primaryName}
                  </h3>
                  {activeChemicalPopup.chemical.preferredName && normalizeLookupText(activeChemicalPopup.chemical.preferredName) !== normalizeLookupText(activeChemicalPopup.chemical.primaryName) && (
                    <p className="mt-1 text-[11px] font-medium text-on-surface-variant normal-case">
                      {activeChemicalPopup.chemical.preferredName}
                    </p>
                  )}
                  {activeChemicalPopup.chemical.synonyms && activeChemicalPopup.chemical.synonyms.length > 0 && (
                    <div className="mt-1 text-[10px] font-medium text-on-surface-muted normal-case">
                      Also: {isExpandedChemical ? (
                        <>
                          {activeChemicalPopup.chemical.synonyms.join(', ')}
                          <button
                            type="button"
                            onClick={() => toggleExpandedChemical(activeChemicalPopup.chemical.primaryName)}
                            className="ml-1 font-semibold text-primary hover:underline cursor-pointer"
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
                              className="ml-1 font-semibold text-primary hover:underline cursor-pointer"
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
                  className="rounded-lg border border-border px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-on-surface-muted transition-colors hover:border-outline hover:text-on-surface-variant cursor-pointer"
                >
                  Close
                </button>
              </div>

              {/* Molecular Structure */}
              <div className="mt-3 min-h-[19rem] rounded-xl border border-border bg-surface-c/70 px-3 py-4">
                {activeChemicalPopup.chemical.smiles ? (
                  <div className="relative flex min-h-[17rem] items-center justify-center">
                    {isRenderingChemicalStructure && (
                      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 rounded-lg bg-background/75 backdrop-blur-[1px]">
                        <div className="h-6 w-6 animate-spin rounded-full border-2 border-teal-200 border-t-teal-500" />
                        <p className="text-xs font-medium text-on-surface-variant">Rendering structure…</p>
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
                      <div className="text-xs text-on-surface-muted">No structure</div>
                    )}
                  </div>
                ) : (
                  <div className="flex min-h-[17rem] items-center justify-center text-xs text-on-surface-muted">No structure</div>
                )}
              </div>

              <div className="mt-4 space-y-2 border-t border-border pt-3 text-[10px] text-on-surface-variant">
                {activeChemicalPopup.chemical.molecularFormula && (
                  <div className="flex items-start gap-2">
                    <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-on-surface-muted">Formula</span>
                    <span className="min-w-0 break-words font-mono font-medium">{activeChemicalPopup.chemical.molecularFormula}</span>
                  </div>
                )}
                {activeChemicalPopup.chemical.inchikey && (
                  <div className="flex items-start gap-2">
                    <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-on-surface-muted">InChIKey</span>
                    <span className="min-w-0 break-words font-mono font-medium break-all">{activeChemicalPopup.chemical.inchikey}</span>
                  </div>
                )}
                {(activeChemicalPopup.chemical.sourceDb || activeChemicalPopup.chemical.sourceUrl) && (
                  <div className="flex items-start gap-2">
                    <span className="w-16 shrink-0 font-semibold uppercase tracking-wider text-on-surface-muted">Source</span>
                    {activeChemicalPopup.chemical.sourceUrl ? (
                      <a
                        href={activeChemicalPopup.chemical.sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="min-w-0 break-words font-medium text-primary hover:underline"
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
                <div className="animate-spin rounded-full h-6 w-6 border-2 border-border border-t-teal-500 mx-auto mb-3" />
                <p className="text-xs text-on-surface-muted">Fetching abstract from alternative sources...</p>
              </div>
            </div>
          )}
        </main>

        {/* Right Sidebar: Entity Groups */}
        <aside className="w-full lg:w-[330px] sticky top-[88px] h-fit flex flex-col gap-3.5 shrink-0 z-30">
          {!isExtracted ? (
            <div style={{
              display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center",
              gap: 16, padding: "32px 24px",
              border: "1px solid var(--border)", borderRadius: 16,
              background: "#FFFFFF"
            }}>
              <span style={{
                width: 44, height: 44, borderRadius: 12,
                display: "grid", placeItems: "center",
                background: "#c8f3fa",
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
                  background: "#c8f3fa", color: "var(--on-surface)",
                  border: "none", cursor: isExtracting ? "wait" : "pointer",
                  fontSize: 14, fontWeight: 600, letterSpacing: ".01em",
                  transition: "opacity .15s",
                  opacity: isExtracting ? 0.7 : 1
                }}
                onMouseEnter={(e) => { if (!isExtracting) e.currentTarget.style.opacity = "0.9"; }}
                onMouseLeave={(e) => { if (!isExtracting) e.currentTarget.style.opacity = "1"; }}
              >
                {isExtracting ? (
                  <>
                    <div className="animate-spin rounded-full h-3.5 w-3.5 border-2 border-current/30 border-t-current mr-2" />
                    Extracting Terms...
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
                  display: "inline-flex", gap: 2,
                  padding: 5, borderRadius: 999,
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
                          height: 34, borderRadius: 999,
                          width: expanded ? "auto" : 38,
                          padding: expanded ? "0 14px" : 0,
                          background: isActive ? "#FFFFFF" : "transparent",
                          boxShadow: isActive ? "0 1px 2px rgba(0,0,0,.08)" : "none",
                          border: "none", cursor: "pointer",
                          fontSize: 13, fontWeight: isActive ? 600 : 500,
                          color: isActive ? "var(--on-surface)" : "var(--on-surface-variant)",
                          whiteSpace: "nowrap", overflow: "hidden",
                          transition: "width .22s ease, padding .22s ease, background .15s, color .15s"
                        }}>
                          {expanded ? <span>{label}</span> : <Icon size={16} />}
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
                    title="Export options" 
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
                      animation: "fadeUp .16s ease"
                    }}>
                      <button 
                        className="export-opt" 
                        onClick={() => {
                          setExportOpen(false);
                          triggerExportCsv();
                        }}
                      >
                        <span>Export CSV</span>
                      </button>
                      <button 
                        className="export-opt" 
                        onClick={() => {
                          setExportOpen(false);
                          triggerExportGraph();
                        }}
                      >
                        <span>Export Graph</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Accordion or KnowledgeGraph */}
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
                          className="ent-cat"
                          onClick={empty ? undefined : () => {
                            toggleEntityGroup(group.label);
                          }}
                          style={{
                            display: "flex", alignItems: "center", gap: 12,
                            width: "100%", padding: "13px 4px",
                            background: "transparent", border: "none",
                            cursor: empty ? "default" : "pointer",
                            opacity: empty ? 0.5 : 1
                          }}
                        >
                          {/* Left category accent dot bar */}
                          <span style={{
                            width: 4, height: 20, borderRadius: 999,
                            background: accentColor, flexShrink: 0,
                            opacity: empty ? 0.3 : 1
                          }} />
                          
                          <span style={{
                            flex: 1, textAlign: "left", fontSize: 14, fontWeight: 600,
                            color: isExpanded ? accentColor : "var(--on-surface)",
                            transition: "color .15s",
                            textTransform: "capitalize"
                          }}>
                            {group.label.toLowerCase()}
                          </span>

                          {/* Hover caret transitions slot */}
                          <span style={{ flexShrink: 0 }}>
                            {empty ? (
                              <span style={{ fontSize: 14, color: "var(--on-surface-variant)" }}>–</span>
                            ) : (
                              <span style={{ position: "relative", display: "inline-grid", placeItems: "center", minWidth: 22, height: 16 }}>
                                <span className="ent-count" style={{
                                  fontSize: 13.5, fontWeight: 600, color: "var(--on-surface-variant)",
                                  opacity: isExpanded ? 0 : 1
                                }}>{group.termCount}</span>
                                <span className={"ent-caret" + (isExpanded ? " is-open" : "")}
                                  style={{
                                    position: "absolute", inset: 0, margin: "auto",
                                    transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
                                    transition: "transform .2s ease, opacity .15s ease",
                                    fontSize: 9, color: "var(--on-surface-variant)",
                                    display: "inline-grid", placeItems: "center",
                                    fontWeight: "bold"
                                  }}
                                >
                                  ▼
                                </span>
                              </span>
                            )}
                          </span>
                        </button>

                        {/* Values expanded container */}
                        {isExpanded && !empty && (
                          <div style={{ padding: "0 4px 12px 16px", display: "flex", flexDirection: "column", gap: 2 }}>
                            {group.visibleItems.map((ent, eIdx) => {
                              const name = ent.text;
                              const isNavigated = nav && nav.name.toLowerCase() === name.toLowerCase();

                              return (
                                <div 
                                  key={eIdx}
                                  onClick={() => activateEntity(name)}
                                  title={`Go to "${name}" in the paper`}
                                  className="ent-row"
                                  style={{
                                    display: "flex", alignItems: "center", gap: 10,
                                    padding: "6px 10px", borderRadius: 7, cursor: "pointer",
                                    transition: "background .12s"
                                  }}
                                  onMouseEnter={(e) => {
                                    e.currentTarget.style.background = "var(--surface-low)";
                                    pulseEntity(name);
                                  }}
                                  onMouseLeave={(e) => {
                                    e.currentTarget.style.background = "transparent";
                                    clearPulse();
                                  }}
                                >
                                  {/* Item dot */}
                                  <span style={{ width: 7, height: 7, borderRadius: 999, background: accentColor, flexShrink: 0 }} />
                                  
                                  {/* Item name (italicized for Species label) */}
                                  <span style={{
                                    flex: 1, fontSize: 13.5,
                                    color: "var(--on-surface)",
                                    fontStyle: group.label === "SPECIES" ? "italic" : "normal"
                                  }}>
                                    {name}
                                  </span>

                                  {/* Stepper pager navigator on active item, or item count otherwise */}
                                  {isNavigated ? (
                                    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }} onClick={(e) => e.stopPropagation()}>
                                      <button onClick={() => gotoMention(nav.idx - 1, nav.total)} title="Previous" style={miniNavStyle}>◀</button>
                                      <span className="mono font-semibold" style={{ fontSize: 11.5, color: "var(--on-surface)", minWidth: 28, textAlign: "center" }}>{nav.idx + 1}/{nav.total}</span>
                                      <button onClick={() => gotoMention(nav.idx + 1, nav.total)} title="Next" style={miniNavStyle}>▶</button>
                                    </span>
                                  ) : (
                                    <span className="mono" style={{ fontSize: 12, color: "var(--on-surface-variant)" }}>
                                      {ent.count}
                                    </span>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="flex-1 min-h-[400px]">
                  <KnowledgeGraph 
                    entities={entities} 
                    paperIdentifier={paperIdentifier}
                    entityConfig={ENTITY_GROUP_CONFIG}
                  />
                </div>
              )}
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
