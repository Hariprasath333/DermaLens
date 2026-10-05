import { useEffect, useRef, useState } from "react";

/**
 * Fetches an image URL via `fetch()` with custom headers and returns a blob URL.
 *
 * This is required because ngrok free-tier injects a browser warning interstitial
 * page on direct `<img src>` requests. By fetching with the
 * `ngrok-skip-browser-warning` header, we bypass the interstitial and get the
 * actual image bytes, which are then served as a local blob URL.
 *
 * For non-ngrok URLs (localhost, blob:, data:) the original URL is returned as-is.
 */

// Module-level cache so switching tabs doesn't re-fetch the same images.
const blobCache = new Map<string, string>();

export function useProxiedImage(url: string | undefined): string | undefined {
  const [blobUrl, setBlobUrl] = useState<string | undefined>(() => {
    if (!url) return undefined;
    if (blobCache.has(url)) return blobCache.get(url);
    const needsProxy = url.includes(".ngrok-free.") || url.includes(".ngrok.");
    return needsProxy ? undefined : url;
  });
  const urlRef = useRef(url);

  useEffect(() => {
    urlRef.current = url;

    if (!url) {
      setBlobUrl(undefined);
      return;
    }

    // Non-ngrok URLs can be used directly.
    const needsProxy = url.includes(".ngrok-free.") || url.includes(".ngrok.");
    if (!needsProxy) {
      setBlobUrl(url);
      return;
    }

    // Already cached? Use it.
    if (blobCache.has(url)) {
      setBlobUrl(blobCache.get(url));
      return;
    }

    let cancelled = false;

    const fetchImage = async (attempt = 0): Promise<void> => {
      try {
        const res = await fetch(url, {
          headers: {
            "ngrok-skip-browser-warning": "true",
            "Accept": "image/*",
          },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const contentType = res.headers.get("content-type") || "";
        if (!contentType.startsWith("image/")) {
          // ngrok returned the interstitial HTML page instead of the image.
          // Retry once after a short delay — the first request sometimes
          // primes the session.
          if (attempt < 2) {
            await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
            if (!cancelled) return fetchImage(attempt + 1);
            return;
          }
          throw new Error(`Expected image, got ${contentType}`);
        }

        const blob = await res.blob();
        if (cancelled) return;
        const objectUrl = URL.createObjectURL(blob);
        blobCache.set(url, objectUrl);
        setBlobUrl(objectUrl);
      } catch {
        // Last resort — pass the raw URL (browser might show interstitial
        // or the user has already accepted it in a previous tab).
        if (!cancelled) setBlobUrl(url);
      }
    };

    fetchImage();

    return () => {
      cancelled = true;
      // Don't revoke — cached blob URLs should persist across re-mounts.
    };
  }, [url]);

  return blobUrl;
}
