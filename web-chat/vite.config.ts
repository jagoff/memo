import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const env = loadEnv("", process.cwd(), "");
const BACKEND = env.SYNAPSE_DASHBOARD_URL ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
});
