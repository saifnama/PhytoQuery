import { createFileRoute } from '@tanstack/react-router';
import MyPapersPage from '../features/papers/MyPapersPage';

export const Route = createFileRoute('/mypapers')({
  component: MyPapersPage,
});
