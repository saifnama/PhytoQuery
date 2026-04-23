import React, { useEffect, useRef, useState } from 'react';
import { Graph, DownloadSimple } from '@phosphor-icons/react';
import type { Entity } from '../../types';

// Declare vis to avoid TS errors
declare global {
  interface Window {
    vis: any;
  }
}

interface KnowledgeGraphProps {
  entities: Entity[];
  paperIdentifier?: { type: string; value: string };
  entityConfig: Record<string, { accentVar: string }>;
}

export const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ entities, paperIdentifier, entityConfig }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<any>(null);
  const dataRef = useRef<{ nodes: any; edges: any }>({ nodes: null, edges: null });
  const [isVisLoaded, setIsVisLoaded] = useState(!!window.vis);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeLegend, setActiveLegend] = useState<{ label: string, color: string }[]>([]);

  // Load vis-network dynamically
  useEffect(() => {
    if (window.vis) {
      setIsVisLoaded(true);
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js';
    script.onload = () => setIsVisLoaded(true);
    document.head.appendChild(script);
  }, []);

  useEffect(() => {
    if (!isVisLoaded || !containerRef.current || entities.length === 0 || !paperIdentifier) return;

    // Build Nodes & Edges
    const nodes = new window.vis.DataSet();
    const edges = new window.vis.DataSet();

    // Central Node (Paper)
    const centralNodeId = `paper-${paperIdentifier.value}`;
    nodes.add({
      id: centralNodeId,
      label: `DOI\n${paperIdentifier.value}`,
      shape: 'dot',
      size: 30,
      color: {
        background: '#E1FBF1', 
        border: '#6EE7B7', // Mint green border to match
        borderWidth: 3,
        highlight: { background: '#D1FAE5', border: '#34D399' }
      },
      font: { color: '#1e293b', face: 'Inter', size: 14, bold: true, vadjust: -5 },
      title: `Type: ${paperIdentifier.type.toUpperCase()}\nIdentifier: ${paperIdentifier.value}`,
      shadow: true,
    });

    // We only want unique canonical entities to avoid clutter, and we count their occurrences
    const uniqueEntities = new Map<string, Entity & { count: number }>();
    entities.forEach(e => {
      // Use text + label as unique key to prevent merging identical terms from different categories
      const key = `${e.label}-${e.text.toLowerCase()}`;
      if (!uniqueEntities.has(key)) {
        uniqueEntities.set(key, { ...e, count: 1 });
      } else {
        const existing = uniqueEntities.get(key)!;
        existing.count += 1;
      }
    });

    // Helper to mix a color with white to lighten it, keeping it 100% opaque
    const lightenRgb = (rgbString: string, factor: number) => {
      const parts = rgbString.split(' ').map(Number);
      if (parts.length !== 3 || parts.some(isNaN)) return rgbString; // fallback
      const [r, g, b] = parts;
      const newR = Math.round(r + (255 - r) * factor);
      const newG = Math.round(g + (255 - g) * factor);
      const newB = Math.round(b + (255 - b) * factor);
      return `rgb(${newR}, ${newG}, ${newB})`;
    };

    const legendMap = new Map<string, string>();

    // Add Entity Nodes
    uniqueEntities.forEach((ent, key) => {
      const config = entityConfig[ent.label.toUpperCase()];
      
      let bgColor = '#cbd5e1'; // Default slate-300 lightened
      let solidColor = '#94a3b8';

      if (config?.accentVar) {
        const rootStyle = getComputedStyle(document.documentElement);
        const rgbResolved = rootStyle.getPropertyValue(config.accentVar + '-rgb').trim();
        if (rgbResolved) {
          // Mix with 40% white to lighten it while keeping it completely solid/opaque
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
      
      legendMap.set(ent.label.toUpperCase(), solidColor);

      nodes.add({
        id: key,
        label: ent.text,
        shape: 'dot',
        size: Math.max(10, Math.min(26, 10 + (ent.count || 1) * 2)), 
        color: {
          background: bgColor,
          border: solidColor, 
          borderWidth: 1.5,
          highlight: { background: bgColor, border: '#1e293b' },
          hover: { background: bgColor, border: '#1e293b' }
        },
        font: { color: '#334155', face: 'Inter', size: Math.max(11, Math.min(14, 9 + ent.count)), vadjust: 2 },
        title: `Type: ${ent.label.toUpperCase()}\nEntity: ${ent.text}\nMentions: ${ent.count || 1}`
      });

      // Connect to central node
      edges.add({
        from: centralNodeId,
        to: key,
        color: { color: '#DBDBDB', highlight: '#969696' },
        width: 1.5
      });
    });

    setActiveLegend(Array.from(legendMap.entries()).map(([label, color]) => ({ label, color })));

    const data = { nodes, edges };
    dataRef.current = data;

    const options = {
      physics: {
        solver: "forceAtlas2Based",
        forceAtlas2Based: {
          gravitationalConstant: -100, // strong repulsion
          centralGravity: 0.015, // weak pull to center
          springLength: 200, // long springs
          springConstant: 0.04,
          damping: 0.4,
          avoidOverlap: 0.6 // strongly push away overlapping nodes
        },
        stabilization: { enabled: true, iterations: 300 }
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        zoomView: true,
        dragView: true
      }
    };

    networkRef.current = new window.vis.Network(containerRef.current, data, options);

    // Zoom out slightly upon initial stabilization
    networkRef.current.once("stabilizationIterationsDone", () => {
      const currentScale = networkRef.current.getScale();
      networkRef.current.moveTo({ scale: currentScale * 0.8 });
    });

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [isVisLoaded, entities, paperIdentifier, entityConfig]);

  // Handle resize when toggling fullscreen
  useEffect(() => {
    if (networkRef.current) {
      setTimeout(() => {
        networkRef.current.redraw();
        networkRef.current.fit({ animation: false });
        
        setTimeout(() => {
          const currentScale = networkRef.current.getScale();
          networkRef.current.moveTo({ 
            scale: currentScale * 0.85, 
            animation: { duration: 300, easingFunction: 'easeInOutQuad' } 
          });
        }, 10);
      }, 50);
    }
  }, [isFullscreen]);

  const downloadHTML = () => {
    if (!dataRef.current.nodes || !dataRef.current.edges) return;
    
    const nodesJSON = JSON.stringify(dataRef.current.nodes.get());
    const edgesJSON = JSON.stringify(dataRef.current.edges.get());
    
    // Build legend HTML
    const legendItemsHTML = activeLegend.map(item => `
      <div class="legend-item">
        <div class="legend-color" style="background-color: ${item.color}"></div>
        <div class="legend-label">${item.label}</div>
      </div>
    `).join('');
    
    const htmlContent = `<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Graph - ${paperIdentifier?.value || 'Export'}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
      body { margin:0; padding:0; overflow:hidden; font-family: 'Inter', sans-serif; background: #f8fafc; }
      #mynetwork { width:100vw; height:100vh; }
      .header { position: absolute; top: 16px; left: 16px; background: white; padding: 12px 16px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; pointer-events: none; }
      .title { font-size: 14px; font-weight: bold; color: #1e293b; margin: 0; }
      .subtitle { font-size: 11px; color: #64748b; margin-top: 4px; }
      .legend { position: absolute; top: 80px; left: 16px; background: rgba(255,255,255,0.9); backdrop-filter: blur(4px); padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); pointer-events: none; max-width: 200px; }
      .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
      .legend-item:last-child { margin-bottom: 0; }
      .legend-color { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
      .legend-label { font-size: 10px; font-weight: 500; color: #334155; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    </style>
</head>
<body>
<div class="header">
  <h1 class="title">Graph View</h1>
  <div class="subtitle">Entities linked to ${paperIdentifier?.value || 'Document'}</div>
</div>
${activeLegend.length > 0 ? `
<div class="legend">
  ${legendItemsHTML}
</div>
` : ''}
<div id="mynetwork"></div>
<script>
    const container = document.getElementById('mynetwork');
    const data = { nodes: ${nodesJSON}, edges: ${edgesJSON} };
    const options = {
      physics: {
        solver: "forceAtlas2Based",
        forceAtlas2Based: { gravitationalConstant: -100, centralGravity: 0.015, springLength: 200, springConstant: 0.04, damping: 0.4, avoidOverlap: 0.6 },
      }
    };
    const network = new vis.Network(container, data, options);
    network.once("stabilizationIterationsDone", () => {
      network.moveTo({ scale: network.getScale() * 0.8 });
    });
</script>
</body>
</html>`;

    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `GraphView-${paperIdentifier?.value?.replace(/[/\\?%*:|"<>]/g, '-') || 'export'}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (entities.length === 0 || !paperIdentifier) {
    return null; // Don't show if no entities are extracted
  }

  return (
    <div className="mt-8 flex flex-col">
      <h2 className="text-[14px] font-semibold text-slate-900 font-display mb-3 px-2">
        Graph View
      </h2>
      
      {/* Container switches between normal box and fullscreen modal overlay */}
      <div className={isFullscreen 
        ? "fixed inset-0 z-[100] bg-slate-900/30 backdrop-blur-sm flex items-center justify-center p-4 sm:p-8 transition-all duration-200"
        : "relative bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm flex flex-col transition-all duration-200"
      }>
        
        {/* Backdrop click-to-close */}
        {isFullscreen && (
          <div className="absolute inset-0 cursor-pointer" onClick={() => setIsFullscreen(false)} />
        )}
        
        {/* Main Card */}
        <div className={isFullscreen 
          ? "relative bg-white w-full max-w-5xl h-[85vh] rounded-2xl shadow-2xl border border-slate-200 flex flex-col overflow-hidden z-10" 
          : "relative w-full flex flex-col"
        }>
          
          {/* Action Buttons Overlay */}
          <div className="absolute top-3 right-3 z-20 flex gap-2">
            {isFullscreen && (
              <button 
                onClick={downloadHTML}
                className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer focus:outline-none"
                title="Download Graph as HTML"
              >
                <DownloadSimple weight="bold" size={18} />
              </button>
            )}
            <button 
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer focus:outline-none"
              title={isFullscreen ? "Close fullscreen" : "Expand to fullscreen"}
            >
              <Graph weight="regular" size={20} />
            </button>
          </div>
          
          {/* Legend Overlay - Only shown when fullscreen */}
          {isFullscreen && activeLegend.length > 0 && (
            <div className="absolute bottom-4 left-4 z-10 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg p-3 shadow-sm max-w-[200px] pointer-events-none origin-bottom-left">
              <div className="flex flex-col gap-2">
                {activeLegend.map((item, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: item.color }} />
                    <span className="text-[11px] font-medium text-slate-700 truncate" title={item.label}>
                      {item.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          <div ref={containerRef} className={`w-full ${isFullscreen ? 'h-full flex-1' : 'h-[300px]'} cursor-grab active:cursor-grabbing bg-transparent`} />
        </div>
      </div>
    </div>
  );
};
