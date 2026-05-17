import { createFileRoute } from '@tanstack/react-router';
import RagPage from '../features/chat/RagPage';

export const Route = createFileRoute('/chat')({
  component: RagPage,
});
