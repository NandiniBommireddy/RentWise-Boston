import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      // The RAG backend runs as a separate Python process (uvicorn on :8000).
      // Proxying keeps the frontend same-origin, so no CORS in dev.
      '/api': {
        target: process.env.RENTWISE_API ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
