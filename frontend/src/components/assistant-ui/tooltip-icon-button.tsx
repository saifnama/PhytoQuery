/**
 * TooltipIconButton — Button + Tooltip composition for icon-only triggers.
 *
 * Wraps shadcn's <Button> in a Radix Tooltip so hover/focus shows a styled
 * tooltip instead of relying on HTML `title=""`. Accepts every Button prop
 * plus a required `tooltip` string and optional `side`.
 *
 * Works under assistant-ui's `asChild` pattern (e.g.
 *   <ActionBarPrimitive.Copy asChild>
 *     <TooltipIconButton tooltip="Copy">...</TooltipIconButton>
 *   </ActionBarPrimitive.Copy>
 * ) because all extra props from the parent primitive are spread onto the
 * inner Button.
 */

import { type ComponentProps, type FC, type ReactNode } from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Button } from '@/components/ui/button';

type Side = 'top' | 'right' | 'bottom' | 'left';

interface TooltipIconButtonProps extends ComponentProps<typeof Button> {
  tooltip: string;
  side?: Side;
  children: ReactNode;
}

export const TooltipIconButton: FC<TooltipIconButtonProps> = ({
  tooltip,
  side = 'bottom',
  children,
  ...props
}) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button {...props}>
          {children}
          <span className="sr-only">{tooltip}</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side={side}>{tooltip}</TooltipContent>
    </Tooltip>
  );
};
