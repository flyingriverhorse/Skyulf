import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/',
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: path.resolve(__dirname, '../../static/ml_canvas'),
    emptyOutDir: true,
    chunkSizeWarningLimit: 10000, // Increase limit for Plotly
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-plotly': ['plotly.js-gl3d-dist-min', 'react-plotly.js'],
          'vendor-charts': ['recharts', 'chart.js', 'react-chartjs-2'],
          'vendor-flow': ['@xyflow/react'],
          'vendor-utils': ['html-to-image', 'lucide-react', 'axios', 'dagre']
        }
      }
    }
  },
  server: {
    proxy: {
      '/data/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ml-workflow': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Realtime job-event invalidator socket (`jobEventsSocket.ts`) connects
      // to `ws://<origin>/ws/jobs`; without `ws: true` here, Vite's dev
      // server proxy never forwards the upgrade request to the backend, so
      // the socket fails immediately with "closed before the connection is
      // established" on every page load in local dev.
      '/ws': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      }
    }
  }
})
