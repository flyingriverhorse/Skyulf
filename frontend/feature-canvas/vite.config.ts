// @ts-nocheck
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: '.',
  base: '/static/feature_canvas/',
  define: {
    'process.env': JSON.stringify({ NODE_ENV: 'production' }),
  },
  appType: 'custom',
  build: {
    lib: {
      entry: resolve(__dirname, 'src/main.tsx'),
      name: 'FeatureCanvasApp',
      fileName: () => 'feature-canvas.js',
      formats: ['iife'],
    },
    outDir: resolve(__dirname, '../../static/feature_canvas'),
    emptyOutDir: true,
    assetsDir: '.',
    sourcemap: true,
    rollupOptions: {
      output: {
        entryFileNames: 'feature-canvas.js',
        chunkFileNames: 'feature-canvas-[name].js',
        assetFileNames: (assetInfo) => {
          if (assetInfo.name && assetInfo.name.endsWith('.css')) {
            return 'feature-canvas.css';
          }
          return 'feature-canvas-[name][extname]';
        }
      }
    }
  }
});
