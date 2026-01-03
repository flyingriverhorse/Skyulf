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
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name][extname]',
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-plotly': ['plotly.js-dist-min', 'react-plotly.js'],
          'vendor-charts': ['recharts', 'chart.js', 'react-chartjs-2'],
          'vendor-flow': ['@xyflow/react'],
          'vendor-utils': ['html2canvas', 'lucide-react', 'axios', 'dagre']
        }
      }
    }
  },
  server: {
    proxy: {
      '/data': {
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
      }
    }
  }
})
