import { createFileRoute } from '@tanstack/react-router';
import AnalysePage from '../features/papers/AnalysePage';

export const Route = createFileRoute('/analyse')({
  component: AnalysePage,
});
