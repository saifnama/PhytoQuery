/**
 * Standalone offline HTML export for knowledge graphs.
 * vis-network is inlined from the locally installed bundle (no CDN, no network).
 * The bundle is loaded lazily via `?raw` so it stays out of the main chunk.
 */
export interface GraphLegendItem {
  color: string;
  name: string;
}

const safeJson = (value: unknown) => JSON.stringify(value).replace(/</g, '\\u003c');

// The minified bundle ends with e.g. `export{VN as Network,...}` — the export
// alias is NOT a local binding, so `new Network()` would throw. Resolve the
// real local name for the `Network` export instead.
const findNetworkCtor = (bundle: string): string => {
  const matches = [...bundle.matchAll(/export\s*\{([^}]*)\}/g)];
  const last = matches.length > 0 ? matches[matches.length - 1] : null;
  if (last) {
    for (const part of last[1].split(',')) {
      const seg = part.trim().split(/\s+as\s+/);
      const local = seg[0].trim();
      const alias = (seg[1] ?? seg[0]).trim();
      if (alias === 'Network' && local) return local;
    }
  }
  return 'Network';
};

export async function downloadGraphHtml(opts: {
  nodes: unknown[];
  edges: unknown[];
  filename: string;
  title?: string;
  subtitle?: string;
  legend?: GraphLegendItem[];
}): Promise<void> {
  // ponytail: local bundle inlined — export opens offline, zero CDN.
  // ESM build goes in a module script; constructor name resolved from its export list.
  const { default: visBundle } = await import(
    'vis-network/standalone/esm/vis-network.min.js?raw'
  );
  const safeBundle = (visBundle as string).replace(/<\/script>/g, '<\\/script>');
  const networkCtor = findNetworkCtor(visBundle as string);

  const legendHtml =
    opts.legend && opts.legend.length > 0
      ? `<div class="legend">${opts.legend
          .map(
            (l) =>
              `<div class="legend-item"><div class="legend-color" style="background-color:${l.color}"></div><div class="legend-label">${l.name}</div></div>`,
          )
          .join('')}</div>`
      : '';

  const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>${opts.title || 'Knowledge Graph'}</title>
<style>
body{margin:0;padding:0;overflow:hidden;font-family:Inter,system-ui,sans-serif;background:#f8fafc}
#mynetwork{width:100vw;height:100vh}
.header{position:absolute;top:16px;left:16px;background:white;padding:12px 16px;border-radius:8px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);border:1px solid #e2e8f0;pointer-events:none}
.title{font-size:14px;font-weight:bold;color:#1e293b;margin:0}
.subtitle{font-size:11px;color:#64748b;margin-top:4px}
.legend{position:absolute;top:80px;left:16px;background:rgba(255,255,255,0.9);padding:10px;border-radius:8px;border:1px solid #e2e8f0;pointer-events:none;max-width:200px}
.legend-item{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.legend-color{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.legend-label{font-size:11px;font-weight:500;color:#334155}
</style></head><body>
<div class="header"><h1 class="title">${opts.title || 'Graph View'}</h1>${opts.subtitle ? `<div class="subtitle">${opts.subtitle}</div>` : ''}</div>
${legendHtml}
<div id="mynetwork"></div>
<script type="module">${safeBundle}
const data={nodes:${safeJson(opts.nodes)},edges:${safeJson(opts.edges)}};
const options={physics:{solver:"forceAtlas2Based",forceAtlas2Based:{gravitationalConstant:-100,centralGravity:0.015,springLength:200,springConstant:0.04,damping:0.85,avoidOverlap:0.6},minVelocity:0.75},interaction:{hover:true,tooltipDelay:100}};
const net=new ${networkCtor}(document.getElementById('mynetwork'),data,options);
net.once("stabilizationIterationsDone",()=>net.moveTo({scale:net.getScale()*0.8}));
<\/script></body></html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = opts.filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
