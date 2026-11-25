const configuredApiBase = (import.meta.env.VITE_API_BASE_URL || '').trim();

export function apiEndpoint(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const base = configuredApiBase.replace(/\/+$/, '');
  return `${base}${normalizedPath}`;
}
