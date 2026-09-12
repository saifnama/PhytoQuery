# AGENTS.md — frontend

- Run: `cd frontend && npm run dev` (port 5173, proxies `/api /search /paper /ner /health /doi` to localhost:8000).
- Build/typecheck: `npm run build` (runs `tsr generate && tsc -b && vite build`); `npx tsc -b --noEmit`.
- Routes are file-based (TanStack Router): `src/routes/` + `src/routeTree.gen.ts` (generated, do not hand-edit).
- Conventions: pages in `src/pages/` (target), reusable domain code in `src/features/<domain>/`, primitives in `src/components/ui/` only, API calls in `src/lib/api/` (splitting `lib/api.ts` per Phase 2).
- Brand: `BloomIndex` title in `index.html` vs `PhytoQuery` repo — see `VISION.md` before renaming anything.
