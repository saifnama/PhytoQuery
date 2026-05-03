/**
 * ForceGraph3D — corpus-wide knowledge graph (Chemical / Species / Location).
 * Renders as either react-force-graph-2d or react-force-graph-3d based on a
 * runtime toggle. Both modes share graphData, colors, and d3-force config.
 */

import React, { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { DownloadSimple, Camera, ArrowCounterClockwise, Cube, Square } from '@phosphor-icons/react';
import type { Graph3DData, GraphNode } from '../../types';

const ForceGraph2D = lazy(() => import('react-force-graph-2d'));
const ForceGraph3DComponent = lazy(() => import('react-force-graph-3d'));

interface ForceGraph3DProps {
  data: Graph3DData;
  onNodeClick?: (node: GraphNode) => void;
}

interface LinkObject {
  source: string;
  target: string;
  weight: number;
}

const LABEL_COLORS: Record<string, string> = {
  CHEMICAL: '#2563eb',
  SPECIES:  '#16a34a',
  LOCATION: '#0891b2',
};
const FALLBACK_COLOR = '#64748b';
const getNodeColor = (label: string) => LABEL_COLORS[label] ?? FALLBACK_COLOR;

const buildLabelHTML = (n: GraphNode) => {
  const color = getNodeColor(n.label);
  return `<div style="font-family:Inter,sans-serif;padding:8px 10px;background:#0f172a;color:white;border-radius:8px;border:1px solid rgba(255,255,255,0.15);box-shadow:0 8px 24px rgba(0,0,0,0.5);max-width:220px">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
      <div style="width:10px;height:10px;border-radius:50%;background:${color};flex-shrink:0"></div>
      <b style="font-size:13px;font-weight:600">${n.name}</b>
    </div>
    <div style="font-size:11px;color:rgba(255,255,255,0.6);margin-bottom:2px">${n.label}</div>
    <div style="font-size:11px;color:rgba(255,255,255,0.4)">${n.count} mentions · ${n.paper_count} papers</div>
  </div>`;
};

export const ForceGraph3D: React.FC<ForceGraph3DProps> = ({ data, onNodeClick }) => {
  const [mode, setMode] = useState<'2d' | '3d'>('3d');
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initial fit after warmup whenever data changes or mode toggles
  useEffect(() => {
    if (!data.nodes.length) return;
    const t = setTimeout(() => graphRef.current?.zoomToFit?.(400, 80), 1500);
    return () => clearTimeout(t);
  }, [data, mode]);

  const exportPNG = useCallback(() => {
    const canvas = containerRef.current?.querySelector('canvas') as HTMLCanvasElement | null;
    if (!canvas) return;
    const a = document.createElement('a');
    a.download = `PhytoQuery-${mode}-Graph-${Date.now()}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  }, [mode]);

  const exportHTML = useCallback(() => {
    const nodes = data.nodes;
    const links = data.links;

    const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>PhytoQuery 3D Knowledge Graph</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.0/dist/3d-force-graph.min.js"></script>
  <style>
    body { margin: 0; background: #0f172a; font-family: 'Inter', sans-serif; }
    #graph { width: 100vw; height: 100vh; }
    .header { position: absolute; top: 16px; left: 16px; background: rgba(15,23,42,0.8); backdrop-filter: blur(8px); padding: 12px 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); color: white; z-index: 10; }
    .header h1 { font-size: 15px; font-weight: 700; margin: 0; }
    .header p { font-size: 11px; color: rgba(255,255,255,0.5); margin: 4px 0 0; }
    .legend { position: absolute; bottom: 16px; left: 16px; background: rgba(15,23,42,0.8); backdrop-filter: blur(8px); padding: 10px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); z-index: 10; }
    .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 11px; color: rgba(255,255,255,0.7); }
    .legend-item:last-child { margin-bottom: 0; }
    .legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  </style>
</head>
<body>
  <div class="header">
    <h1>PhytoQuery 3D Knowledge Graph</h1>
    <p>${nodes.length} entities · ${links.length} connections</p>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#2563eb"></div>Chemical</div>
    <div class="legend-item"><div class="legend-dot" style="background:#16a34a"></div>Species</div>
    <div class="legend-item"><div class="legend-dot" style="background:#0891b2"></div>Location</div>
  </div>
  <div id="graph"></div>
  <script>
    const DATA = ${JSON.stringify(data)};
    const COLORS = ${JSON.stringify(LABEL_COLORS)};
    const container = document.getElementById('graph');
    const g = window.ForceGraph3D(container)
      .graphData(DATA)
      .nodeColor(d => COLORS[d.label] || '#64748b')
      .nodeVal(d => Math.max(1, Math.log2((d.count||0)+1)*2.5))
      .linkColor(() => 'rgba(148,163,184,0.25)')
      .linkWidth(d => Math.max(0.3, Math.log2((d.weight||0)+1)*1.2))
      .d3VelocityDecay(0.3)
      .warmupTicks(100)
      .cooldownTicks(300);
    setTimeout(() => g.zoomToFit(400, 80), 1500);
  </script>
</body>
</html>`;

    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PhytoQuery-3D-Graph-${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }, [data]);

  const resetCamera = useCallback(() => graphRef.current?.zoomToFit?.(400, 80), []);

  if (!data.nodes.length) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400 text-sm">
        No entity data available.
      </div>
    );
  }

  // Shared props (canvas/three accessors are silently ignored on the wrong variant)
  const sharedProps = {
    graphData: data,
    nodeLabel: (node: unknown) => buildLabelHTML(node as GraphNode),
    nodeColor: (node: unknown) => getNodeColor((node as GraphNode).label),
    nodeRelSize: 4,
    nodeVal: (node: unknown) => Math.max(1, Math.log2((node as GraphNode).count + 1) * 2.5),
    linkColor: () => 'rgba(148,163,184,0.25)',
    linkWidth: (link: unknown) => Math.max(0.3, Math.log2((link as LinkObject).weight + 1) * 1.2),
    linkDirectionalArrowLength: 2,
    linkDirectionalArrowRelPos: 1,
    linkDirectionalParticles: 2,
    linkDirectionalParticleSpeed: 0.005,
    linkDirectionalParticleWidth: 0.6,
    linkDirectionalParticleColor: () => 'rgba(255,255,255,0.6)',
    d3VelocityDecay: 0.3,
    warmupTicks: 100,
    cooldownTicks: 300,
    onNodeClick: (node: unknown) => {
      if (onNodeClick) onNodeClick(node as GraphNode);
    },
    onNodeHover: (node: unknown) => {
      setHoveredNode(node ? (node as GraphNode) : null);
      if (containerRef.current) {
        containerRef.current.style.cursor = node ? 'pointer' : 'default';
      }
    },
  };

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full">
        <Suspense
          fallback={
            <div className="flex items-center justify-center h-full text-slate-400 text-sm">
              <div className="w-5 h-5 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin mr-3" />
              Loading {mode.toUpperCase()} renderer…
            </div>
          }
        >
          {mode === '3d' ? (
            <ForceGraph3DComponent
              ref={graphRef}
              backgroundColor="#0f172a"
              {...sharedProps}
            />
          ) : (
            <ForceGraph2D
              ref={graphRef}
              backgroundColor="#0f172a"
              {...sharedProps}
            />
          )}
        </Suspense>
      </div>

      {/* Floating controls */}
      <div className="absolute top-3 right-3 flex gap-2 z-10">
        {/* 2D / 3D toggle */}
        <div className="flex bg-slate-800/80 backdrop-blur-sm border border-slate-600 rounded-lg overflow-hidden shadow-sm">
          <button
            onClick={() => setMode('2d')}
            className={`px-2 py-2 text-xs font-medium transition-colors flex items-center gap-1 ${
              mode === '2d' ? 'bg-blue-600 text-white' : 'text-slate-300 hover:bg-slate-700'
            }`}
            title="2D view"
          >
            <Square weight="bold" size={12} /> 2D
          </button>
          <button
            onClick={() => setMode('3d')}
            className={`px-2 py-2 text-xs font-medium transition-colors flex items-center gap-1 ${
              mode === '3d' ? 'bg-blue-600 text-white' : 'text-slate-300 hover:bg-slate-700'
            }`}
            title="3D view"
          >
            <Cube weight="bold" size={12} /> 3D
          </button>
        </div>

        <button
          onClick={resetCamera}
          className="p-2 bg-slate-800/80 backdrop-blur-sm border border-slate-600 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700 transition-colors shadow-sm"
          title="Reset view"
        >
          <ArrowCounterClockwise weight="bold" size={14} />
        </button>
        <button
          onClick={exportPNG}
          className="p-2 bg-slate-800/80 backdrop-blur-sm border border-slate-600 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700 transition-colors shadow-sm"
          title="Export PNG"
        >
          <Camera weight="bold" size={14} />
        </button>
        <button
          onClick={exportHTML}
          className="p-2 bg-slate-800/80 backdrop-blur-sm border border-slate-600 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700 transition-colors shadow-sm"
          title="Export interactive HTML"
        >
          <DownloadSimple weight="bold" size={14} />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-slate-900/80 backdrop-blur-sm border border-slate-700 rounded-lg px-3 py-2 shadow-sm z-10">
        <div className="flex flex-col gap-1.5">
          {[
            { label: 'Chemical', color: '#2563eb' },
            { label: 'Species',  color: '#16a34a' },
            { label: 'Location', color: '#0891b2' },
          ].map(({ label, color }) => (
            <div key={label} className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-[11px] text-slate-300">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Node hover info */}
      {hoveredNode && (
        <div className="absolute bottom-4 left-4 bg-slate-900/80 backdrop-blur-sm border border-slate-700 rounded-lg px-3 py-2 shadow-sm z-10">
          <div className="text-xs font-semibold text-white">{hoveredNode.name}</div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            {hoveredNode.label} · {hoveredNode.count} mentions · {hoveredNode.paper_count} papers
          </div>
        </div>
      )}
    </div>
  );
};
