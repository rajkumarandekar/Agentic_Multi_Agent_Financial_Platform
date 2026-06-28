import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Forward all API calls to the FastAPI backend so the browser never
    // makes a cross-origin request during development.
    proxy: {
      '/agent':     'http://localhost:8000',
      '/ask':       'http://localhost:8000',
      '/upload':    'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/metrics':   'http://localhost:8000',
      '/dashboard': 'http://localhost:8000',
      '/health':    'http://localhost:8000',
    },
  },
})
