import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Pin to the IPv4 loopback explicitly: "localhost" can resolve to ::1 first,
// which silently proxies to whatever else happens to be bound on port 8000's
// IPv6 loopback instead of this project's API.
const API = process.env.MATCHLAB_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": API,
    },
  },
  preview: {
    port: 5173,
    proxy: {
      "/api": API,
    },
  },
});
