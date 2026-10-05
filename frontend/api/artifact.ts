// Vercel Edge Function — proxies /artifacts/* through the bypass header so
// ngrok's free-tier interstitial (ERR_NGROK_6024 "You are about to visit...")
// doesn't get returned in place of the image bytes.
//
// Routing (configured in vercel.json):
//   Browser  /artifacts/CASE/raw.png
//      ↓ rewrite to /api/artifact?p=CASE/raw.png
//   Vercel   runs this function
//      ↓ fetch with ngrok-skip-browser-warning: true
//   ngrok    serves the actual image bytes
//
// Single non-bracketed file with query-param routing — picked up reliably
// by Vercel + Vite without depending on dynamic catch-all path resolution.

export const config = { runtime: "edge" };

const NGROK_BASE = "https://overload-opacity-connector.ngrok-free.dev";

export default async function handler(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const path = url.searchParams.get("p") || "";
  if (!path) {
    return new Response("Missing ?p= parameter", { status: 400 });
  }
  const target = `${NGROK_BASE}/artifacts/${path}`;

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: "GET",
      headers: {
        "ngrok-skip-browser-warning": "true",
        "User-Agent": "LesionIQ-Vercel-Proxy/1.0"
      }
    });
  } catch (err) {
    return new Response(
      `Artifact proxy upstream error: ${err instanceof Error ? err.message : String(err)}`,
      { status: 502, headers: { "Content-Type": "text/plain" } }
    );
  }

  const headers = new Headers();
  ["content-type", "content-length", "last-modified", "etag"].forEach((h) => {
    const v = upstream.headers.get(h);
    if (v) headers.set(h, v);
  });
  headers.set("Cache-Control", "public, max-age=3600, immutable");

  return new Response(upstream.body, {
    status: upstream.status,
    headers
  });
}
