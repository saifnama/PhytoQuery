import axios from 'axios';

/**
 * Pull a readable `detail` message out of an axios error.
 * Blob responses (responseType: 'blob') arrive unparsed, so the raw Blob
 * must be read as text first — otherwise callers only ever see the fallback.
 */
export async function extractErrorDetail(err: unknown, fallback: string): Promise<string> {
  const data =
    typeof err === 'object' && err !== null && 'response' in err
      ? (err as { response?: { data?: unknown } }).response?.data
      : undefined;
  if (!data) return errorMessage(err, fallback);
  if (typeof data === 'object' && !(data instanceof Blob)) {
    return (data as { detail?: string }).detail || fallback;
  }
  try {
    const text = typeof data === 'string' ? data : await (data as Blob).text();
    return (JSON.parse(text) as { detail?: string }).detail || text || fallback;
  } catch {
    return fallback;
  }
}

/** Best-effort message from anything thrown. Matches the old
 * `err?.message || fallback` semantics exactly (only non-empty string
 * messages win; everything else falls back). */
export function errorMessage(err: unknown, fallback: string): string {
  if (typeof err === 'object' && err !== null && 'message' in err) {
    const message = (err as { message?: unknown }).message;
    if (typeof message === 'string' && message) return message;
  }
  return fallback;
}

const API_BASE = ''; // Uses Vite proxy in dev, same origin in production

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000,
  withCredentials: true,
});

export default api;
