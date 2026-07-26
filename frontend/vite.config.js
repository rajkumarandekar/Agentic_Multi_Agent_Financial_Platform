import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Use a single regex rule that matches every backend path so new endpoints
    // are automatically proxied without editing this file.
    // Matches: /agent, /ask, /upload, /documents, /metrics, /dashboard,
    //          /health, /reset, /logs, /system, /chats, /chats/*, etc.
    proxy: {
      '^/(agent|ask|upload|documents|metrics|dashboard|health|reset|logs|system|chats)': {
        target: 'http://localhost:8000',
        changeOrigin: false,
      },
    },
  },
})
