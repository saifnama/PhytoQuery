import * as React from 'react';
import { cn } from '@/lib/utils';

/**
 * Kbd — inline keyboard chip.
 * Used by the search bar to show the ⌘K hint on the right edge of the
 * collapsed pill. Styled with JetBrains Mono, surface-c background and
 * a hairline outline-variant border so it matches the rest of the
 * design system.
 */
export const Kbd: React.FC<React.ComponentProps<'kbd'>> = ({
  className,
  children,
  ...props
}) => (
  <kbd
    className={cn(
      'kbd select-none',
      className,
    )}
    {...props}
  >
    {children}
  </kbd>
);

export default Kbd;
