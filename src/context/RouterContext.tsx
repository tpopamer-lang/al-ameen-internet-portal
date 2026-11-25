import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { AppRoutePath, ROUTE_META, resolveRoutePath } from '../config/routes';

interface RouterContextValue {
  currentPath: AppRoutePath;
  currentSearch: string;
  navigate: (path: string) => void;
}

const RouterContext = createContext<RouterContextValue | null>(null);

// Vite replaces BASE_URL at build time. It is "/" locally and
// "/al-ameen-internet-portal/" on this GitHub Pages deployment.
const BASE_PATH = import.meta.env.BASE_URL.replace(/\/$/, '');

function getAppPathname(pathname: string = window.location.pathname): string {
  if (BASE_PATH && (pathname === BASE_PATH || pathname.startsWith(`${BASE_PATH}/`))) {
    return pathname.slice(BASE_PATH.length) || '/';
  }
  return pathname || '/';
}

function toAppUrl(pathname: string): string {
  const normalized = pathname.startsWith('/') ? pathname : `/${pathname}`;
  return `${BASE_PATH}${normalized}` || '/';
}

function getInitialRoute(): AppRoutePath {
  const redirectedPath = new URLSearchParams(window.location.search).get('p');
  return resolveRoutePath(redirectedPath || getAppPathname());
}

export function RouterProvider({ children }: { children: React.ReactNode }) {
  const [currentPath, setCurrentPath] = useState<AppRoutePath>(getInitialRoute);
  const [currentSearch, setCurrentSearch] = useState<string>(() => {
    const params = new URLSearchParams(window.location.search);
    params.delete('p');
    const query = params.toString();
    return query ? `?${query}` : '';
  });

  const syncMeta = useCallback((path: AppRoutePath) => {
    const meta = ROUTE_META[path];
    document.title = meta.title;
    const descEl = document.querySelector('meta[name="description"]');
    if (descEl) descEl.setAttribute('content', meta.description);
    const ogTitle = document.querySelector('meta[property="og:title"]');
    if (ogTitle) ogTitle.setAttribute('content', meta.title);
    const ogDesc = document.querySelector('meta[property="og:description"]');
    if (ogDesc) ogDesc.setAttribute('content', meta.description);
  }, []);

  useEffect(() => {
    const url = new URL(window.location.href);
    const redirectedPath = url.searchParams.get('p');
    if (redirectedPath) url.searchParams.delete('p');

    const initialResolved = resolveRoutePath(redirectedPath || getAppPathname());
    const canonicalPath = toAppUrl(initialResolved);
    const nextSearch = url.search;
    if (window.location.pathname !== canonicalPath || redirectedPath) {
      window.history.replaceState({}, '', `${canonicalPath}${nextSearch}`);
    }
    syncMeta(initialResolved);
  }, [syncMeta]);

  useEffect(() => {
    const handlePopState = () => {
      const resolved = resolveRoutePath(getAppPathname());
      setCurrentPath(resolved);
      setCurrentSearch(window.location.search || '');
      syncMeta(resolved);
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [syncMeta]);

  const navigate = useCallback(
    (targetPath: string) => {
      const qIdx = targetPath.indexOf('?');
      const rawPathname = qIdx >= 0 ? targetPath.slice(0, qIdx) : targetPath;
      const rawSearch = qIdx >= 0 ? targetPath.slice(qIdx) : '';
      const resolved = resolveRoutePath(getAppPathname(rawPathname));
      const nextUrl = `${toAppUrl(resolved)}${rawSearch}`;
      if (`${window.location.pathname}${window.location.search}` !== nextUrl) {
        window.history.pushState({}, '', nextUrl);
      }
      setCurrentPath(resolved);
      setCurrentSearch(rawSearch);
      syncMeta(resolved);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    },
    [syncMeta]
  );

  return (
    <RouterContext.Provider value={{ currentPath, currentSearch, navigate }}>
      {children}
    </RouterContext.Provider>
  );
}

export function useRouter(): RouterContextValue {
  const ctx = useContext(RouterContext);
  if (!ctx) {
    throw new Error('useRouter must be used inside RouterProvider');
  }
  return ctx;
}

interface AppLinkProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  to: string;
}

export function AppLink({ to, onClick, children, ...rest }: AppLinkProps) {
  const { navigate } = useRouter();
  const queryIndex = to.indexOf('?');
  const pathname = queryIndex >= 0 ? to.slice(0, queryIndex) : to;
  const search = queryIndex >= 0 ? to.slice(queryIndex) : '';
  const href = `${toAppUrl(resolveRoutePath(getAppPathname(pathname)))}${search}`;

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    if (onClick) onClick(e);
    if (e.defaultPrevented) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) {
      return;
    }
    e.preventDefault();
    navigate(to);
  };

  return (
    <a href={href} onClick={handleClick} {...rest}>
      {children}
    </a>
  );
}
