import * as React from 'react';

/**
 * TulipLogo — the brand mark used in the Header.
 * Hand-coded SVG with two linearGradients:
 *   - `tulipPetal`  pink-to-magenta   (#F06292 → #C2185B)
 *   - `tulipStem`   mint-to-green     (#81C784 → #388E3C)
 *
 * Two overlapping petals + a single teardrop leaf. The whole mark is
 * gradient-only, so it sits cleanly on any background without a
 * bounding box or background fill.
 */
export const TulipLogo: React.FC<{ size?: number; className?: string }> = ({
  size = 32,
  className,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 64 64"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="PhytoQuery"
  >
    <defs>
      <linearGradient id="tulipPetal" x1="32" y1="6" x2="32" y2="46" gradientUnits="userSpaceOnUse">
        <stop offset="0" stopColor="#F06292" />
        <stop offset="1" stopColor="#C2185B" />
      </linearGradient>
      <linearGradient id="tulipStem" x1="20" y1="32" x2="48" y2="58" gradientUnits="userSpaceOnUse">
        <stop offset="0" stopColor="#81C784" />
        <stop offset="1" stopColor="#388E3C" />
      </linearGradient>
    </defs>
    {/* Stem (back leaf) */}
    <path
      d="M28 50 C 28 40, 36 32, 44 28 L 46 32 C 38 36, 32 44, 32 52 Z"
      fill="url(#tulipStem)"
    />
    {/* Petal — left half */}
    <path
      d="M14 22 C 14 12, 22 6, 32 6 C 28 16, 26 24, 26 32 C 22 30, 18 28, 14 22 Z"
      fill="url(#tulipPetal)"
    />
    {/* Petal — right half */}
    <path
      d="M50 22 C 50 12, 42 6, 32 6 C 36 16, 38 24, 38 32 C 42 30, 46 28, 50 22 Z"
      fill="url(#tulipPetal)"
    />
    {/* Center petal — narrower, deeper */}
    <path
      d="M32 6 C 28 14, 27 22, 28 34 L 36 34 C 37 22, 36 14, 32 6 Z"
      fill="url(#tulipPetal)"
      opacity="0.95"
    />
    {/* Base / calyx */}
    <ellipse cx="32" cy="34" rx="7" ry="3" fill="#00796B" opacity="0.85" />
  </svg>
);

export default TulipLogo;
