# PhytoQuery Frontend

React + TypeScript + Vite frontend for PhytoQuery. The paper reader includes chemical popups that render molecular structures client-side from SMILES strings using the local `smiles-drawer` package.

## Development

```bash
bun install
bun run dev
```

`bun install` installs all local frontend dependencies, including `smiles-drawer`.

## Build

```bash
bun run build
```

The build output goes to `frontend/dist/`.

## Tech Stack

- React 19 + TypeScript
- Vite
- Tailwind CSS
- Phosphor Icons
- React Router
- smiles-drawer

## Notes

- Chemical popup molecule rendering is frontend-only and uses a live SVG render path via `smiles-drawer`.
- There is no CDN dependency for molecule rendering in the popup.
