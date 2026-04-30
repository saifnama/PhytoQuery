import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/ner': {
        target: 'http://localhost:8000',
        timeout: 600000,
      },
      '/rag': {
        target: 'http://localhost:8000',
        timeout: 600000,
      },
      '/health': 'http://localhost:8000',
      '/static': 'http://localhost:8000',
      '/api': {
        target: 'http://localhost:8000',
        timeout: 600000,
      },
      '/doi': 'http://localhost:8000',
    },
  },
})
