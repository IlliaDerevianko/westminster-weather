import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // The bill directory is the home page; "/?bill_id=…" still opens a bill
  async redirects() {
    return [
      {
        source: "/",
        missing: [{ type: "query", key: "bill_id" }],
        destination: "/bills",
        permanent: false,
      },
    ];
  },
  // The browser calls /api/v1/* on this site and Next forwards it to the backend:
  // locally the dev API on port 8000, on Vercel the URL in BACKEND_URL (read at build time)
  async rewrites() {
    const backend = (process.env.BACKEND_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
    return [
      {
        source: "/api/v1/:path*",
        destination: `${backend}/api/v1/:path*`,
      },
    ];
  },
};

// A production build must not overwrite assets used by a running dev server.
export default (phase) => ({
  ...nextConfig,
  distDir: phase === PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next",
});
