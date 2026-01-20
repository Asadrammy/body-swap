import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // Backend runs on port 8000 (from configs/default.yaml)
        changeOrigin: true,
        secure: false,
        ws: true, // Enable WebSocket proxying if needed
      },
    },
  },
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
})

