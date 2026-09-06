/**
 * KnowledgeGraph — per-paper graph using vis-network (npm, bundled).
 * Includes: hover/click highlight focus, React-rendered tooltip, search-to-focus,
 * layout switcher (force/hierarchical/circular), filter pills, double-click
 * collapse, auto-disable physics after stabilization.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CornersOut, CornersIn, DownloadSimple, ArrowCounterClockwise } from '@phosphor-icons/react';
import type { Entity } from '../../types';
import { downloadGraphHtml } from '../../utils/exportGraphHtml';

// vis-network's TS surface is overly strict for our loose option objects.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type VisModule = any;

interface PaperIdentifier {
  type: string;
  value: string;
}

interface KnowledgeGraphProps {
  entities: Entity[];
  paperIdentifier?: PaperIdentifier;
  paperIdentifiers?: PaperIdentifier[];
  entityConfig: Record<string, { accentVar: string }>;
  /** Map of entity key -> array of paper values this entity appears in (for compare mode) */
  entityPaperMap?: Record<string, string[]>;
}

// ─── vis-network options (single force layout, physics stays alive) ──────
//   • edges:   straight (no curves), DASHED by default, light grey
//   • nodes:   no black border on highlight; halo (glow shadow) on selected
//              applied dynamically via DataSet.update in click handler
function buildOptions() {
  return {
    nodes: {
      shape: 'dot',
      borderWidth: 0,
    },
    edges: {
      smooth: false, // straight lines
      color: {
        color: '#e2e8f0', // very light grey at rest
        highlight: '#64748b', // medium grey when connected to selected
        hover: '#64748b',
        opacity: 1,
      },
      width: 0.8, // thin idle
      hoverWidth: 1.5, // slightly bolder on hover/click
      selectionWidth: 1.5,
    },
    interaction: {
      hover: true,
      tooltipDelay: 9999, // suppress vis-network's built-in tooltip; we render our own
      zoomView: true,
      dragView: true,
    },
    layout: { randomSeed: 42 },
    // Physics tuned to match the v4 HTML's feel:
    //   springLength 130 ≈ "30% of smaller canvas dim" at typical view sizes —
    //   keeps entities close to the paper instead of pinned to edges.
    physics: {
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -100,
        centralGravity: 0.015,
        springLength: 130,
        springConstant: 0.04,
        damping: 0.85,
        avoidOverlap: 0.6,
      },
      stabilization: { enabled: true, iterations: 300 },
      minVelocity: 0.3, // sim runs longer → bounce on filter toggle is visible
    },
  };
}

// Pretty-print an entity label (e.g. PLANT_PART → "Plant Part")
function prettyType(label: string): string {
  return label
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({
  entities,
  paperIdentifier,
  paperIdentifiers,
  entityConfig,
  entityPaperMap,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const networkRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodesDS = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const edgesDS = useRef<any>(null);

  const [vis, setVis] = useState<VisModule | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [search, setSearch] = useState('');
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; nodeId: string } | null>(null);

  // Refs for closures inside vis-network event handlers (avoid stale state)
  const selectedIdRef = useRef<string | null>(null);
  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);
  const activeTypesRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    activeTypesRef.current = activeTypes;
  }, [activeTypes]);

  const papers = useMemo(
    () => paperIdentifiers || (paperIdentifier ? [paperIdentifier] : []),
    [paperIdentifiers, paperIdentifier]
  );

  // ── Build node/edge data + per-type metadata ─────────────────────────────
  const { nodesData, edgesData, typeColors, nodeInfo } = useMemo(() => {
    if (entities.length === 0 || papers.length === 0) {
      return {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodesData: [] as any[],
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        edgesData: [] as any[],
        typeColors: new Map<string, { name: string; color: string; count: number }>(),
        nodeInfo: new Map<
          string,
          { type: string; name: string; freq: number; color: string; isPaper: boolean }
        >(),
      };
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const nodesData: any[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const edgesData: any[] = [];
    const typeColors = new Map<string, { name: string; color: string; count: number }>();
    const nodeInfo = new Map<
      string,
      { type: string; name: string; freq: number; color: string; isPaper: boolean }
    >();

    // Paper anchor node(s)
    const paperNodeIds: string[] = [];
    papers.forEach((p) => {
      const nodeId = `paper-${p.value}`;
      paperNodeIds.push(nodeId);
      const isSingle = papers.length === 1;
      const labelText = `DOI\n${p.value}`;
      nodesData.push({
        id: nodeId,
        label: showLabels ? labelText : '',
        _label: labelText,
        _isPaper: true,
        _type: 'PAPER',
        _color: '#10B981', // for halo on selection
        shape: 'dot',
        size: isSingle ? 34 : 28,
        color: {
          background: '#E1FBF1',
          border: '#6EE7B7',
          // No colour change on highlight/hover — selection is shown via the
          // halo shadow we set in the click handler, not a different border.
          highlight: { background: '#E1FBF1', border: '#6EE7B7' },
          hover: { background: '#E1FBF1', border: '#6EE7B7' },
        },
        borderWidth: isSingle ? 4 : 3,
        font: {
          color: '#1e293b',
          face: 'Google Sans, Inter, sans-serif',
          size: isSingle ? 16 : 13,
          bold: true,
          vadjust: -5,
        },
        shadow: { enabled: true, color: 'rgba(0,0,0,0.08)', size: 8, x: 0, y: 2 },
      });
      nodeInfo.set(nodeId, {
        type: 'Paper',
        name: p.value,
        freq: 0,
        color: '#10B981',
        isPaper: true,
      });
    });

    // Aggregate entity counts
    const uniqueEntities = new Map<string, Entity & { count: number }>();
    entities.forEach((e) => {
      const key = `${e.label}-${e.text.toLowerCase()}`;
      const incomingCount = (e as Entity & { count?: number }).count || 1;
      if (!uniqueEntities.has(key)) {
        uniqueEntities.set(key, { ...e, count: incomingCount });
      } else {
        uniqueEntities.get(key)!.count += incomingCount;
      }
    });

    // Helper: lighten an "r g b" string by mixing with white
    const lightenRgb = (rgbString: string, factor: number) => {
      const parts = rgbString.split(' ').map(Number);
      if (parts.length !== 3 || parts.some(isNaN)) return rgbString;
      const [r, g, b] = parts;
      return `rgb(${Math.round(r + (255 - r) * factor)}, ${Math.round(g + (255 - g) * factor)}, ${Math.round(b + (255 - b) * factor)})`;
    };

    uniqueEntities.forEach((ent, key) => {
      const labelKey = ent.label.toUpperCase();
      const config = entityConfig[labelKey];

      let bgColor = '#cbd5e1';
      let solidColor = '#94a3b8';

      if (config?.accentVar) {
        const rootStyle = getComputedStyle(document.documentElement);
        const rgbResolved = rootStyle.getPropertyValue(`${config.accentVar}-rgb`).trim();
        if (rgbResolved) {
          bgColor = lightenRgb(rgbResolved, 0.4);
          solidColor = `rgb(${rgbResolved.split(' ').join(',')})`;
        } else {
          const resolved = rootStyle.getPropertyValue(config.accentVar).trim();
          if (resolved) {
            bgColor = resolved;
            solidColor = resolved;
          }
        }
      }

      // Per-type metadata for legend / pills
      const existing = typeColors.get(labelKey);
      if (existing) {
        existing.count += 1;
      } else {
        typeColors.set(labelKey, { name: prettyType(labelKey), color: solidColor, count: 1 });
      }

      // Frequency-driven sizing — sqrt with a small multiplier keeps the
      // spread visible but compact. Range 8–22 px, font 10–14 px.
      const count = ent.count || 1;
      const freqSize = Math.max(8, Math.min(22, 8 + Math.sqrt(count) * 3.5));
      const freqFontSize = Math.max(10, Math.min(14, 9 + Math.log2(count + 1) * 1.5));

      nodesData.push({
        id: key,
        label: showLabels ? ent.text : '',
        _label: ent.text,
        _isPaper: false,
        _type: labelKey,
        _color: solidColor, // for halo on selection
        shape: 'dot',
        size: freqSize,
        color: {
          background: bgColor,
          border: solidColor,
          // No black border on highlight — selection shown via halo only
          highlight: { background: bgColor, border: solidColor },
          hover: { background: bgColor, border: solidColor },
        },
        borderWidth: ent.count && ent.count > 5 ? 2 : 1.5,
        font: { color: '#334155', face: 'Google Sans, Inter, sans-serif', size: freqFontSize, vadjust: 2 },
      });
      nodeInfo.set(key, {
        type: prettyType(labelKey),
        name: ent.text,
        freq: ent.count || 1,
        color: solidColor,
        isPaper: false,
      });

      // Edges
      const papersForEntity = entityPaperMap?.[key];
      if (papersForEntity && papersForEntity.length > 0) {
        papersForEntity.forEach((paperValue) => {
          edgesData.push({
            id: `e-${paperValue}-${key}`,
            from: `paper-${paperValue}`,
            to: key,
          });
        });
      } else {
        edgesData.push({ id: `e-${paperNodeIds[0]}-${key}`, from: paperNodeIds[0], to: key });
      }
    });

    return { nodesData, edgesData, typeColors, nodeInfo };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entities, paperIdentifier, paperIdentifiers, entityConfig, entityPaperMap]);

  // Initialize / reset activeTypes whenever the type set changes (new paper)
  const [prevTypeColors, setPrevTypeColors] = useState<Map<string, unknown> | null>(null);
  if (prevTypeColors !== typeColors) {
    setPrevTypeColors(typeColors);
    setActiveTypes(new Set(typeColors.keys()));
  }

  // ── Lazy-load vis-network bundle on mount ─────────────────────────────
  useEffect(() => {
    let cancelled = false;
    import('vis-network/standalone').then((mod) => {
      if (!cancelled) setVis(mod);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // ── CSS reset for vis-network's tooltip wrapper (we don't use it, but
  //    keep the style clean if it ever flashes).
  useEffect(() => {
    if (document.getElementById('vis-tooltip-reset')) return;
    const style = document.createElement('style');
    style.id = 'vis-tooltip-reset';
    style.textContent = `
      .vis-tooltip {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        pointer-events: none !important;
      }
    `;
    document.head.appendChild(style);
  }, []);

  // ── Mount vis-network ─────────────────────────────────────────────────
  useEffect(() => {
    if (!vis || !containerRef.current || nodesData.length === 0) return;

    const data = {
      nodes: new vis.DataSet(nodesData),
      edges: new vis.DataSet(edgesData),
    };
    nodesDS.current = data.nodes;
    edgesDS.current = data.edges;

    const network = new vis.Network(containerRef.current, data, buildOptions());
    networkRef.current = network;

    // Slight zoom-out once the layout settles. Physics stays enabled so the
    // user can drag nodes and watch the springs respond.
    network.on('stabilizationIterationsDone', () => {
      const scale = network.getScale();
      network.moveTo({ scale: scale * 0.8 });
    });

    // Edge styling — thin light grey at rest, medium grey + slightly bolder
    // when connected to a hovered/selected node.
    const lightConnectedEdges = (nodeId: string) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      edgesDS.current.get().forEach((e: any) => {
        const isConn = e.from === nodeId || e.to === nodeId;
        edgesDS.current.update({
          id: e.id,
          width: isConn ? 1.5 : 0.8,
          color: { color: isConn ? '#64748b' : '#e2e8f0', opacity: 1 },
        });
      });
    };
    const restoreAllEdges = () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      edgesDS.current.get().forEach((e: any) => {
        edgesDS.current.update({
          id: e.id,
          width: 0.8,
          color: { color: '#e2e8f0', opacity: 1 },
        });
      });
    };

    // Halo helpers — colored shadow on the selected node, default shadow on others
    const applyHalo = (nodeId: string) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      nodesData.forEach((n: any) => {
        if (n.id === nodeId) {
          nodesDS.current.update({
            id: n.id,
            shadow: {
              enabled: true,
              color: n._color || '#10B981',
              size: 22,
              x: 0,
              y: 0,
            },
          });
        } else {
          nodesDS.current.update({
            id: n.id,
            shadow: n._isPaper
              ? { enabled: true, color: 'rgba(0,0,0,0.08)', size: 8, x: 0, y: 2 }
              : false,
          });
        }
      });
    };
    const clearHalo = () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      nodesData.forEach((n: any) => {
        nodesDS.current.update({
          id: n.id,
          shadow: n._isPaper
            ? { enabled: true, color: 'rgba(0,0,0,0.08)', size: 8, x: 0, y: 2 }
            : false,
        });
      });
    };

    // Hover: dim non-neighbours + light up edges + show React tooltip
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    network.on('hoverNode', (params: any) => {
      const nodeId = params.node;
      if (selectedIdRef.current === null) {
        const connected = new Set<string>(network.getConnectedNodes(nodeId));
        connected.add(nodeId);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodesDS.current.get().forEach((n: any) => {
          nodesDS.current.update({ id: n.id, opacity: connected.has(n.id) ? 1 : 0.12 });
        });
        lightConnectedEdges(nodeId);
      }
      const pos = params.pointer.DOM;
      setTooltip({ x: pos.x, y: pos.y, nodeId });
    });

    network.on('blurNode', () => {
      if (selectedIdRef.current === null) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodesDS.current.get().forEach((n: any) => nodesDS.current.update({ id: n.id, opacity: 1 }));
        restoreAllEdges();
      }
      setTooltip(null);
    });

    // Click: select → HIDE all non-connected nodes (vis-network auto-hides
    //   their edges too); click empty → restore visibility per current filter.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    network.on('click', (params: any) => {
      if (params.nodes.length === 0) {
        setSelectedId(null);
        // Restore everything to opacity 1 + visible according to filter
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodesData.forEach((n: any) => {
          nodesDS.current.update({
            id: n.id,
            opacity: 1,
            hidden: n._isPaper ? false : !activeTypesRef.current.has(n._type),
          });
        });
        restoreAllEdges();
        clearHalo();
        return;
      }
      const nodeId = params.nodes[0];
      const connected = new Set<string>(network.getConnectedNodes(nodeId));
      connected.add(nodeId);
      setSelectedId(nodeId);
      // Hide everything except selected + its connected neighbours.
      // (vis-network automatically hides edges whose endpoint is hidden)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      nodesData.forEach((n: any) => {
        nodesDS.current.update({
          id: n.id,
          opacity: 1,
          hidden: !connected.has(n.id),
        });
      });
      lightConnectedEdges(nodeId);
      applyHalo(nodeId);
    });

    // Double-click: collapse / expand neighbours (skip paper nodes)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    network.on('doubleClick', (params: any) => {
      if (params.nodes.length === 0) return;
      const nodeId: string = params.nodes[0];
      if (nodeId.startsWith('paper-')) return;
      const neighbours: string[] = network.getConnectedNodes(nodeId);
      neighbours.forEach((id) => {
        if (id.startsWith('paper-')) return; // never hide paper
        const node = nodesDS.current.get(id);
        if (!node) return;
        nodesDS.current.update({ id, hidden: !node.hidden });
      });
    });

    return () => {
      network.destroy();
      networkRef.current = null;
      nodesDS.current = null;
      edgesDS.current = null;
    };
  }, [vis, nodesData, edgesData]);

  // ── Sync labels toggle ────────────────────────────────────────────────
  useEffect(() => {
    if (!nodesDS.current) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    nodesData.forEach((n: any) => {
      nodesDS.current.update({ id: n.id, label: showLabels ? n._label : '' });
    });
  }, [showLabels, nodesData]);

  // ── Sync type filter ──────────────────────────────────────────────────
  // Animate the bounce: toggle hidden, perturb newly-revealed nodes, then
  // call startSimulation() — vis-network animates frame-by-frame at 60 fps
  // until the system reaches minVelocity. (stabilize(N) runs offscreen and
  // shows no animation, which is why it didn't bounce visibly.)
  useEffect(() => {
    if (!nodesDS.current || !networkRef.current) return;
    const network = networkRef.current;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    nodesData.forEach((n: any) => {
      if (n._isPaper) return;
      const willHide = !activeTypes.has(n._type);
      const current = nodesDS.current.get(n.id);
      const wasHidden = !!current?.hidden;

      nodesDS.current.update({ id: n.id, hidden: willHide });

      // Re-showing: nudge position so physics has motion to resolve. Without
      // a kick, a node sitting at its old equilibrium has nothing to do and
      // the bounce is invisible.
      if (wasHidden && !willHide) {
        const positions = network.getPositions([n.id]);
        const pos = positions[n.id];
        if (pos) {
          const dx = (Math.random() - 0.5) * 80;
          const dy = (Math.random() - 0.5) * 80;
          network.moveNode(n.id, pos.x + dx, pos.y + dy);
        }
      }
    });

    network.startSimulation();
  }, [activeTypes, nodesData]);

  // ── Search: focus matching node, dim others ───────────────────────────
  useEffect(() => {
    if (!networkRef.current || !nodesDS.current) return;
    if (!search.trim()) {
      // Restore opacity (unless a selection is active)
      if (selectedIdRef.current === null) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodesDS.current.get().forEach((n: any) => nodesDS.current.update({ id: n.id, opacity: 1 }));
      }
      return;
    }
    const q = search.toLowerCase();
    const matches: string[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    nodesData.forEach((n: any) => {
      if (n._label && n._label.toLowerCase().includes(q)) matches.push(n.id);
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    nodesDS.current.get().forEach((n: any) => {
      nodesDS.current.update({ id: n.id, opacity: matches.includes(n.id) ? 1 : 0.1 });
    });
    if (matches.length > 0) {
      networkRef.current.focus(matches[0], {
        scale: 1.6,
        animation: { duration: 600, easingFunction: 'easeInOutQuad' },
      });
    }
  }, [search, nodesData]);

  // ── Refit after entering/exiting fullscreen (post-paint sizing) ───────
  useEffect(() => {
    if (!networkRef.current) return;
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        const net = networkRef.current;
        if (!net) return;
        net.redraw();
        net.fit({ animation: false });
        const scale = net.getScale();
        net.moveTo({
          scale: scale * 0.85,
          animation: { duration: 300, easingFunction: 'easeInOutQuad' },
        });
      });
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [isFullscreen]);

  // ── True fullscreen via the Fullscreen API (Esc handled by browser) ───
  // ponytail: native API — no z-index fights, no portal, no remount.
  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) {
      void document.exitFullscreen().catch(() => {});
    } else {
      void wrapperRef.current?.requestFullscreen().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  // ── Helpers ───────────────────────────────────────────────────────────
  const toggleType = useCallback((type: string) => {
    setActiveTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }, []);

  const resetLayout = useCallback(() => {
    if (!networkRef.current) return;
    // Re-shake the layout (physics is already enabled — just kick the simulation)
    networkRef.current.stabilize(150);
    setSearch('');
    setSelectedId(null);
    setActiveTypes(new Set(typeColors.keys()));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    nodesDS.current?.get().forEach((n: any) => {
      nodesDS.current.update({ id: n.id, opacity: 1, hidden: false });
    });
  }, [typeColors]);

  // ── HTML export (standalone, offline — no CDN) ─────────────────────────
  const downloadHTML = useCallback(() => {
    if (!nodesDS.current || !edgesDS.current) return;
    const paperLabel =
      papers.length > 1 ? `${papers.length} Papers` : (paperIdentifier?.value || 'Document');
    const safeLabel =
      papers.length > 1
        ? `compare-${papers.length}-papers`
        : paperIdentifier?.value?.replace(/[/\\?%*:|"<>]/g, '-') || 'export';
    void downloadGraphHtml({
      nodes: nodesDS.current.get(),
      edges: edgesDS.current.get(),
      filename: `GraphView-${safeLabel}.html`,
      title: `Knowledge Graph - ${paperLabel}`,
      subtitle: `Entities linked to ${paperLabel}`,
      legend: [...typeColors.entries()].map(([, info]) => ({
        color: info.color,
        name: info.name,
      })),
    });
  }, [typeColors, papers, paperIdentifier]);

  const hasPaper = paperIdentifier || (paperIdentifiers && paperIdentifiers.length > 0);
  if (entities.length === 0 || !hasPaper) return null;

  const presentTypes = [...typeColors.entries()];
  const tooltipInfo = tooltip ? nodeInfo.get(tooltip.nodeId) : null;

  return (
    <div className="flex flex-col">

      <div
        ref={wrapperRef}
        className={
          isFullscreen
            ? 'bg-background w-screen h-[100dvh] flex flex-col overflow-hidden'
            : 'relative bg-background border border-border rounded-xl overflow-hidden shadow-sm flex flex-col transition-all duration-200'
        }
      >
        <div
          className={
            isFullscreen
              ? 'relative w-full h-full flex flex-col overflow-hidden'
              : 'relative w-full flex flex-col'
          }
        >
          {/* Top control bar */}
          <div className="absolute top-3 right-3 z-20 flex gap-2 items-center">
            {isFullscreen && (
              <>
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search node…"
                  className="text-[12px] px-3 py-1.5 rounded-lg border border-border bg-background/90 text-on-surface-variant outline-none focus:border-on-surface-muted w-36"
                  style={{ fontFamily: 'var(--font-google-sans)' }}
                />
                <button
                  type="button"
                  onClick={resetLayout}
                  className="p-2 text-on-surface-muted hover:text-on-surface-variant hover:bg-surface-c rounded-lg transition-colors cursor-pointer focus:outline-none flex items-center justify-center"
                  aria-label="Reset layout"
                >
                  <ArrowCounterClockwise weight="bold" size={18} />
                </button>
                <button
                  type="button"
                  onClick={() => setShowLabels((v) => !v)}
                  className={`text-[12px] font-medium px-3 py-1.5 rounded-lg border transition-colors cursor-pointer ${
                    showLabels
                      ? 'bg-on-surface text-background border-on-surface'
                      : 'bg-background text-on-surface-variant border-border hover:bg-surface-c'
                  }`}
                  style={{ fontFamily: 'var(--font-google-sans)' }}
                  aria-label="Toggle labels"
                >
                  Labels
                </button>
              </>
            )}
            <button
              type="button"
              onClick={downloadHTML}
              className="p-2 text-on-surface-muted hover:text-on-surface-variant hover:bg-surface-c rounded-lg transition-colors cursor-pointer focus:outline-none flex items-center justify-center"
              aria-label="Download"
            >
              <DownloadSimple weight="bold" size={20} />
            </button>
            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-2 text-on-surface-muted hover:text-on-surface-variant hover:bg-surface-c rounded-lg transition-colors cursor-pointer focus:outline-none flex items-center justify-center"
              aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            >
              {isFullscreen ? (
                <CornersIn weight="bold" size={20} />
              ) : (
                <CornersOut weight="bold" size={20} />
              )}
            </button>
          </div>

          {/* Legend with counts — borderless, fullscreen only */}
          {isFullscreen && presentTypes.length > 0 && (
            <div
              className="absolute top-4 left-4 z-10 min-w-[240px] p-2"
              style={{ fontFamily: 'var(--font-google-sans)' }}
            >
              <div className="text-[12px] font-bold uppercase tracking-[0.12em] text-on-surface-variant mb-2 px-2">
                Entities
              </div>
              <div className="space-y-0.5">
                {presentTypes.map(([type, info]) => {
                  const on = activeTypes.has(type);
                  return (
                    <div
                      key={type}
                      onClick={() => toggleType(type)}
                      className="flex items-center gap-3 px-2 py-2 rounded-md cursor-pointer transition-all hover:bg-surface-c/70"
                      style={{ opacity: on ? 1 : 0.3 }}
                    >
                      <div
                        className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{ background: info.color }}
                      />
                      <span className="text-[14px] font-medium text-on-surface flex-1 truncate">
                        {info.name}
                      </span>
                      <span className="text-[13px] font-semibold text-on-surface-variant tabular-nums">
                        {info.count}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* React tooltip — fires from network's hoverNode event */}
          {tooltip && tooltipInfo && (
            <div
              className="absolute pointer-events-none z-30"
              style={{
                left: tooltip.x + 14,
                top: tooltip.y - 10,
                background: '#fff',
                border: '1px solid #e2e2e2',
                borderRadius: 8,
                padding: '9px 12px',
                boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
                minWidth: 155,
                fontFamily: 'var(--font-google-sans), Inter, sans-serif',
              }}
            >
              {!tooltipInfo.isPaper && (
                <div
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    color: tooltipInfo.color,
                    marginBottom: 3,
                    fontFamily: 'var(--font-google-sans)',
                  }}
                >
                  {tooltipInfo.type}
                </div>
              )}
              <div
                style={{
                  fontSize: 13.5,
                  fontWeight: 600,
                  color: '#111',
                  marginBottom: 3,
                  lineHeight: 1.3,
                  fontFamily: 'var(--font-google-sans)',
                }}
              >
                {tooltipInfo.name}
              </div>
              {!tooltipInfo.isPaper && (
                <div style={{ fontFamily: 'var(--font-google-sans)', fontSize: 11, color: '#64748b' }}>
                  frequency · {tooltipInfo.freq}
                </div>
              )}
            </div>
          )}

          {/* Graph container */}
          <div
            ref={containerRef}
            className={`w-full ${isFullscreen ? 'flex-1 min-h-0' : 'h-[300px]'} cursor-grab active:cursor-grabbing bg-transparent`}
          />
        </div>
      </div>
    </div>
  );
};
