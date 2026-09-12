import axios from 'axios';

/**
 * Pull a readable `detail` message out of an axios error.
 * Blob responses (responseType: 'blob') arrive unparsed, so the raw Blob
 * must be read as text first — otherwise callers only ever see the fallback.
 */
export async function extractErrorDetail(err: any, fallback: string): Promise<string> {
  const data = err?.response?.data;
  if (!data) return err?.message || fallback;
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

const API_BASE = ''; // Uses Vite proxy in dev, same origin in production

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000,
  withCredentials: true,
});

export default api;
