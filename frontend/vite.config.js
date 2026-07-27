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
        // Moved off 8000 -- that port had a stuck/orphaned OS-level socket
        // (a phantom listener with no real owning process) that survived
        // every attempt to kill it. Run the backend with `--port 8001` to
        // match.
        target: 'http://localhost:8001',
        changeOrigin: false,
      },
    },
  },
})
