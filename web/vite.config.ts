import { defineConfig } from "vite";

// index.html is currently self-contained; this config exists so the UI can
// grow into a real frontend app. /api calls proxy to the FastAPI backend.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
