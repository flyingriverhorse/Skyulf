import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { DatasetService } from '../core/api/datasets';
import { EDAService } from '../core/api/eda';
import { VariableCard } from '../components/eda/VariableCard';
import { JobsHistoryModal } from '../components/eda/JobsHistoryModal';
import { CorrelationHeatmap } from '../components/eda/CorrelationHeatmap';
import { DistributionChart } from '../components/eda/DistributionChart';
import { CanvasScatterPlot } from '../components/eda/CanvasScatterPlot';
import { ThreeDScatterPlot } from '../components/eda/ThreeDScatterPlot';
import { Loader2, Play, RefreshCw, AlertCircle, BarChart2, Clock, X, Lightbulb, ScatterChart as ScatterIcon, CheckCircle, Info, Map, Calendar, List, AlertTriangle, Box, Eye, EyeOff, Download, Plus, Filter } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, Legend, LineChart, Line, ComposedChart, Scatter } from 'recharts';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import Plotly from 'plotly.js-dist-min';

// Color palette for PCA clusters
const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#FFBB28', '#FF8042', '#a4de6c', '#d0ed57'];

export const EDAPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<number | null>(() => {
    const id = searchParams.get('dataset_id');
    return id ? Number(id) : null;
  });
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('sample');
  const [targetCol, setTargetCol] = useState<string>('');
  const [excludedCols, setExcludedCols] = useState<string[]>([]);
  const [filters, setFilters] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [selectedVariable, setSelectedVariable] = useState<any>(null);
  
  // Manual Filter State
  const [showFilterForm, setShowFilterForm] = useState(false);
  const [newFilterCol, setNewFilterCol] = useState('');
  const [newFilterOp, setNewFilterOp] = useState('==');
  const [newFilterVal, setNewFilterVal] = useState('');

  // Scatter Plot State
  const [scatterX, setScatterX] = useState<string>('');
  const [scatterY, setScatterY] = useState<string>('');
  const [scatterZ, setScatterZ] = useState<string>('');
  const [scatterColor, setScatterColor] = useState<string>('');
  const [is3D, setIs3D] = useState(false);
  const [isPCA3D, setIsPCA3D] = useState(false);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      setReport(null); // Clear previous report to show loader
      loadReport(selectedDataset);
      loadHistory(selectedDataset);
    } else {
      setReport(null);
      setHistory([]);
    }
  }, [selectedDataset]);

  // Poll for status if pending
  useEffect(() => {
    let interval: any;
    if (report && report.status === 'PENDING') {
      interval = setInterval(() => {
        if (selectedDataset) {
            loadReport(selectedDataset, true);
            loadHistory(selectedDataset);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [report, selectedDataset]);

  const loadDatasets = async () => {
    try {
      const data = await DatasetService.getUsable();
      setDatasets(data);
      if (data.length > 0 && !selectedDataset) {
        setSelectedDataset(Number(data[0].id));
      }
    } catch (err) {
      console.error("Failed to load datasets", err);
    }
  };

  const loadHistory = async (id: number) => {
    try {
        const data = await EDAService.getHistory(id);
        setHistory(data);
    } catch (err) {
        console.error("Failed to load history", err);
    }
  };

  const loadReport = async (id: number, silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const data = await EDAService.getLatestReport(id);
      setReport(data);
    } catch (err: any) {
      if (err.response && err.response.status === 404) {
        setReport(null); // No report yet
      } else {
        // Only show error if not silent (polling)
        if (!silent) setError("Failed to load report");
      }
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const loadSpecificReport = async (reportId: number) => {
    setLoading(true);
    try {
        const data = await EDAService.getReport(reportId);
        setReport(data);
    } catch (err) {
        setError("Failed to load report");
    } finally {
        setLoading(false);
    }
  };

  const runAnalysis = async (overrideExcluded?: string[] | any, overrideFilters?: any[]) => {
    const actualExcluded = Array.isArray(overrideExcluded) ? overrideExcluded : excludedCols;
    const actualFilters = Array.isArray(overrideFilters) ? overrideFilters : filters;

    if (!selectedDataset) return;
    setAnalyzing(true);
    try {
      await EDAService.analyze(selectedDataset, targetCol || undefined, actualExcluded, actualFilters);
      // Reload immediately to get the PENDING state
      loadReport(selectedDataset);
      loadHistory(selectedDataset);
    } catch (err) {
      setError("Failed to start analysis");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleAddFilter = (column: string, value: any, operator: string = '==') => {
      // Check if filter already exists to avoid duplicates if needed, 
      // but for now let's allow multiple filters on same col (e.g. range)
      const newFilter = { column, operator, value };
      const newFilters = [...filters, newFilter];
      setFilters(newFilters);
      runAnalysis(undefined, newFilters);
  };

  const handleRemoveFilter = (index: number) => {
      const newFilters = [...filters];
      newFilters.splice(index, 1);
      setFilters(newFilters);
      runAnalysis(undefined, newFilters);
  };

  const handleToggleExclude = (colName: string, exclude: boolean) => {
    const message = exclude 
        ? `Are you sure you want to exclude '${colName}' from the analysis? This will trigger a new analysis.`
        : `Include '${colName}' back in the analysis? This will trigger a new analysis.`;
        
    if (confirm(message)) {
        let newExcluded = [...excludedCols];
        if (exclude) {
            newExcluded.push(colName);
        } else {
            newExcluded = newExcluded.filter(c => c !== colName);
        }
        setExcludedCols(newExcluded);
        runAnalysis(newExcluded);
    }
  };

  // Sync target col and excluded cols from report if available
  useEffect(() => {
    if (report && report.profile_data) {
        if (report.profile_data.target_col) {
            setTargetCol(report.profile_data.target_col);
        }
        if (report.profile_data.excluded_columns) {
            setExcludedCols(report.profile_data.excluded_columns);
        } else {
            setExcludedCols([]);
        }
    }
  }, [report]);

  const InfoTooltip = ({ text }: { text: string }) => (
    <div className="group relative ml-2 inline-flex items-center">
        <Info className="w-4 h-4 text-gray-400 cursor-help" />
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none text-center">
            {text}
            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-800"></div>
        </div>
    </div>
  );

  const downloadChart = async (elementId: string, filename: string, title?: string, subtitle?: string) => {
    const element = document.getElementById(elementId);
    if (!element) return;

    const isDarkMode = document.documentElement.classList.contains('dark');
    const bgColor = isDarkMode ? '#1f2937' : '#ffffff'; // gray-800 vs white
    const textColor = isDarkMode ? '#f3f4f6' : '#111827'; // gray-100 vs gray-900
    const subTextColor = isDarkMode ? '#9ca3af' : '#4b5563'; // gray-400 vs gray-600

    // Helper to draw text on canvas
    const drawText = (ctx: CanvasRenderingContext2D, width: number) => {
        if (!title) return 0;
        ctx.fillStyle = textColor;
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(title, width / 2, 30);
        
        if (subtitle) {
            ctx.fillStyle = subTextColor;
            ctx.font = '12px sans-serif';
            ctx.fillText(subtitle, width / 2, 50);
            return 60; // Height offset
        }
        return 40; // Height offset
    };

    // Check if it's a Plotly chart (3D Scatter)
    const plotlyDiv = element.querySelector('.js-plotly-plot');
    if (plotlyDiv) {
        try {
            // Use Plotly.toImage to get a high-res PNG
            const dataUrl = await Plotly.toImage(plotlyDiv as any, { format: 'png', width: 1200, height: 800 });
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            const width = 1200;
            const height = 800;
            const headerHeight = title ? (subtitle ? 80 : 50) : 0;

            img.onload = () => {
                canvas.width = width;
                canvas.height = height + headerHeight;
                
                if (ctx) {
                    ctx.fillStyle = bgColor;
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw Title
                    if (title) {
                        ctx.fillStyle = textColor;
                        ctx.font = 'bold 24px sans-serif';
                        ctx.textAlign = 'center';
                        ctx.fillText(title, width / 2, 40);
                        
                        if (subtitle) {
                            ctx.fillStyle = subTextColor;
                            ctx.font = '16px sans-serif';
                            ctx.fillText(subtitle, width / 2, 70);
                        }
                    }

                    ctx.drawImage(img, 0, headerHeight);
                    
                    const a = document.createElement('a');
                    a.download = `${filename}.png`;
                    a.href = canvas.toDataURL('image/png');
                    a.click();
                }
            };
            img.src = dataUrl;
        } catch (e) {
            console.error("Plotly download failed", e);
        }
        return;
    }

    // Check if it's an SVG (Recharts)
    const svg = element.querySelector('svg');
    if (svg) {
        const svgData = new XMLSerializer().serializeToString(svg);
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        // Get dimensions
        const width = svg.clientWidth || svg.getBoundingClientRect().width || 800;
        const height = svg.clientHeight || svg.getBoundingClientRect().height || 400;
        const headerHeight = title ? (subtitle ? 60 : 40) : 0;

        img.onload = () => {
            canvas.width = width;
            canvas.height = height + headerHeight;
            
            if (ctx) {
                // Background
                ctx.fillStyle = bgColor;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw Title
                drawText(ctx, width);
                
                // Draw Chart
                ctx.drawImage(img, 0, headerHeight);
                
                const a = document.createElement('a');
                a.download = `${filename}.png`;
                a.href = canvas.toDataURL('image/png');
                a.click();
            }
        };
        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
        return;
    }
    
    // Check if it's a Canvas (3D Scatter, Leaflet, Chart.js)
    // Note: Chart.js and Plotly usually create their own canvas
    const sourceCanvas = element.querySelector('canvas');
    if (sourceCanvas) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        const width = sourceCanvas.width;
        const height = sourceCanvas.height;
        // Scale down if it's high DPI to match text size roughly, or scale text up.
        // Usually sourceCanvas.width is physical pixels.
        // Let's assume we draw text relative to that.
        
        const headerHeight = title ? (subtitle ? 80 : 50) : 0; // Larger offset for high-res canvas

        canvas.width = width;
        canvas.height = height + headerHeight;

        if (ctx) {
            // Background
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw Title (Scale font size based on width)
            if (title) {
                ctx.fillStyle = textColor;
                ctx.font = `bold ${Math.max(16, width / 40)}px sans-serif`;
                ctx.textAlign = 'center';
                ctx.fillText(title, width / 2, headerHeight / 2);
                
                if (subtitle) {
                    ctx.fillStyle = subTextColor;
                    ctx.font = `${Math.max(12, width / 50)}px sans-serif`;
                    ctx.fillText(subtitle, width / 2, headerHeight / 2 + (width / 40));
                }
            }

            // Draw original canvas
            ctx.drawImage(sourceCanvas, 0, headerHeight);

            const a = document.createElement('a');
            a.download = `${filename}.png`;
            a.href = canvas.toDataURL('image/png');
            a.click();
        }
        return;
    }
    
    // Fallback for Correlation Heatmap (HTML Grid)
    // Since we can't easily screenshot HTML without a library, we might be stuck.
    // But wait, CorrelationHeatmap is just divs. 
    // If the user says "correlation doesnt download works", it's because I was looking for SVG/Canvas and found neither.
    // I can't fix this without html2canvas.
    // However, I can try to see if I can construct a canvas from the data directly?
    // That would be re-implementing the heatmap rendering logic.
    // Given the constraints, I will disable download for Correlation Matrix if I can't do it, 
    // OR I can try to find if there is a canvas hidden there.
    // The code for CorrelationHeatmap uses divs. So it won't work.
    // I will remove the download button for Correlation Matrix for now or add a note.
    // actually, I can't easily add html2canvas.
    // I will remove the download button for Correlation Matrix as requested "correlation doesnt download works" implies it's broken.
  };

  const downloadCorrelationHeatmap = () => {
    if (!report?.profile_data?.correlations) return;
    
    const data = report.profile_data.correlations;
    const MAX_COLS = 20;
    const columns = data.columns.slice(0, MAX_COLS);
    const values = data.values.slice(0, MAX_COLS).map((row: number[]) => row.slice(0, MAX_COLS));
    
    const cellSize = 60;
    
    // Calculate dynamic label sizes
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.font = '12px sans-serif';
    let maxLabelWidth = 0;
    columns.forEach((col: string) => {
        const w = ctx.measureText(col).width;
        if (w > maxLabelWidth) maxLabelWidth = w;
    });
    
    // Add padding
    const labelWidth = maxLabelWidth + 40; // Increased padding
    // For rotated labels, the height depends on the length of the text
    // sin(45) * width approx 0.7 * width
    const headerHeight = (maxLabelWidth * 0.7) + 60; // Increased header height
    const titleHeight = 60;
    
    const width = labelWidth + (columns.length * cellSize) + 50; // +50 padding
    const height = headerHeight + (columns.length * cellSize) + titleHeight + 50;

    canvas.width = width;
    canvas.height = height;

    // Background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Title
    ctx.fillStyle = '#111827';
    ctx.font = 'bold 24px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Correlation Matrix', width / 2, 40);

    // Helper for color
    const getColor = (val: number) => {
        if (val === null) return '#f3f4f6';
        const opacity = Math.max(0.2, Math.abs(val));
        if (val > 0) {
            return `rgba(239, 68, 68, ${opacity})`;
        } else {
            return `rgba(59, 130, 246, ${opacity})`;
        }
    };

    ctx.font = '12px sans-serif';
    ctx.textBaseline = 'middle';

    // Draw Grid
    values.forEach((row: number[], i: number) => {
        // Row Label (Right aligned)
        ctx.fillStyle = '#374151';
        ctx.textAlign = 'right';
        ctx.fillText(columns[i], labelWidth - 10, headerHeight + titleHeight + (i * cellSize) + (cellSize/2));

        row.forEach((val: number, j: number) => {
            const x = labelWidth + (j * cellSize);
            const y = headerHeight + titleHeight + (i * cellSize);

            // Cell
            ctx.fillStyle = getColor(val);
            ctx.fillRect(x, y, cellSize - 2, cellSize - 2); // -2 for gap

            // Value
            if (val !== null) {
                ctx.fillStyle = Math.abs(val) > 0.5 ? '#ffffff' : '#000000';
                ctx.textAlign = 'center';
                ctx.fillText(val.toFixed(2), x + (cellSize/2), y + (cellSize/2));
            }
        });
    });

    // Column Labels (Rotated)
    ctx.save();
    columns.forEach((col: string, j: number) => {
        const x = labelWidth + (j * cellSize) + (cellSize/2);
        const y = headerHeight + titleHeight - 10;
        
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = '#374151';
        ctx.textAlign = 'left';
        ctx.fillText(col, 0, 0);
        ctx.rotate(Math.PI / 4);
        ctx.translate(-x, -y);
    });
    ctx.restore();

    // Download
    const link = document.createElement('a');
    link.download = 'correlation-matrix.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const downloadAllInteractions = async () => {
    if (!report?.profile_data?.target_interactions) return;
    
    const interactions = report.profile_data.target_interactions;
    const chartHeight = 400; // Height per chart
    const chartWidth = 800;
    const padding = 40;
    
    // 2 Columns Layout
    const cols = 2;
    const rows = Math.ceil(interactions.length / cols);
    
    const totalWidth = (cols * chartWidth) + ((cols + 1) * padding);
    const totalHeight = (rows * (chartHeight + padding)) + 100; // +100 for main title
    
    const canvas = document.createElement('canvas');
    canvas.width = totalWidth;
    canvas.height = totalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Background
    const isDarkMode = document.documentElement.classList.contains('dark');
    const bgColor = isDarkMode ? '#1f2937' : '#ffffff';
    const textColor = isDarkMode ? '#ffffff' : '#111827';
    const subTextColor = isDarkMode ? '#9ca3af' : '#4b5563';

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Main Title
    ctx.fillStyle = textColor;
    ctx.font = 'bold 24px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Detailed Interactions (Box Plots)', totalWidth / 2, 50);

    // Draw each chart
    for (let i = 0; i < interactions.length; i++) {
        const interaction = interactions[i];
        const elementId = `interaction-chart-${i}`;
        const element = document.getElementById(elementId);
        if (!element) continue;

        const svg = element.querySelector('svg');
        if (!svg) continue;
        
        const row = Math.floor(i / cols);
        const col = i % cols;
        
        const x = padding + (col * (chartWidth + padding));
        const y = 100 + (row * (chartHeight + padding));

        // Title for this chart
        ctx.fillStyle = subTextColor;
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${interaction.feature} vs ${report.profile_data.target_col}`, x + 40, y);
        
        if (interaction.p_value !== undefined) {
            ctx.font = '14px sans-serif';
            ctx.fillStyle = interaction.p_value < 0.05 ? '#059669' : subTextColor;
            ctx.fillText(`ANOVA p: ${Number(interaction.p_value).toExponential(2)}`, x + 40, y + 20);
        }

        // Serialize SVG
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svg);
        const img = new Image();
        const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(svgBlob);

        await new Promise<void>((resolve) => {
            img.onload = () => {
                ctx.drawImage(img, x, y + 30, chartWidth, chartHeight - 50); // Adjust height to fit
                URL.revokeObjectURL(url);
                resolve();
            };
            img.src = url;
        });
    }

    // Download
    const link = document.createElement('a');
    link.download = 'all-interactions.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const renderContent = () => {
    if (loading && !report) {
      return (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-red-500">
          <AlertCircle className="w-16 h-16 mb-4" />
          <p>Error</p>
          <p className="text-sm text-gray-600 mt-2">{error}</p>
          <button
            onClick={() => selectedDataset && loadReport(selectedDataset)}
            className="mt-4 flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </button>
        </div>
      );
    }

    if (!report) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-gray-500">
          <BarChart2 className="w-16 h-16 mb-4 opacity-20" />
          <p className="mb-4">No analysis found for this dataset.</p>
          
          <div className="flex flex-col items-center space-y-4">
            <div className="w-64">
                <input
                type="text"
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                placeholder="Target Column (Optional)"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                />
            </div>
            <button
                onClick={runAnalysis}
                disabled={analyzing}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
                {analyzing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                Run Analysis
            </button>
          </div>
        </div>
      );
    }

    if (report.status === 'PENDING') {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-gray-500">
          <Loader2 className="w-16 h-16 mb-4 animate-spin text-blue-500" />
          <p>Analysis in progress...</p>
          <p className="text-sm text-gray-400">This may take a few moments.</p>
        </div>
      );
    }

    if (report.status === 'FAILED') {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-red-500">
          <AlertCircle className="w-16 h-16 mb-4" />
          <p>Analysis Failed</p>
          <p className="text-sm text-gray-600 mt-2">{report.error_message}</p>
          <button
            onClick={runAnalysis}
            className="mt-4 flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </button>
        </div>
      );
    }

    const profile = report.profile_data;
    if (!profile) return <div>No profile data</div>;

    return (
      <div className="space-y-6">
        {/* Overview Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-500">Rows</div>
            <div className="text-2xl font-bold">{profile.row_count.toLocaleString()}</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-500">Columns</div>
            <div className="text-2xl font-bold">{profile.column_count}</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-500">Missing Cells</div>
            <div className="text-2xl font-bold">{profile.missing_cells_percentage.toFixed(1)}%</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-500">Duplicates</div>
            <div className="text-2xl font-bold">{profile.duplicate_rows}</div>
          </div>
        </div>

        {/* Alerts */}
        {profile.alerts && profile.alerts.length > 0 && (
          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-medium text-amber-800 dark:text-amber-200 flex items-center">
                <AlertCircle className="w-4 h-4 mr-2" />
                Data Quality Alerts ({profile.alerts.length})
              </h3>
            </div>
            <div className="max-h-40 overflow-y-auto pr-2">
                <ul className="space-y-1">
                {profile.alerts.map((alert: any, i: number) => (
                    <li key={i} className="text-sm text-amber-700 dark:text-amber-300 flex items-start">
                    <span className="mr-2">•</span>
                    <span>
                        {alert.column && <span className="font-semibold">{alert.column}: </span>}
                        {alert.message}
                    </span>
                    </li>
                ))}
                </ul>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('sample')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'sample'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Sample Data
            </button>
            <button
              onClick={() => setActiveTab('variables')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'variables'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Variables
            </button>
            <button
              onClick={() => setActiveTab('bivariate')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'bivariate'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Bivariate
            </button>
            <button
              onClick={() => setActiveTab('insights')}
              className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                activeTab === 'insights'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Lightbulb className="w-4 h-4" />
              Smart Insights
            </button>
            {profile.outliers && (
                <button
                onClick={() => setActiveTab('outliers')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                    activeTab === 'outliers'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                >
                <AlertTriangle className="w-4 h-4" />
                Outliers
                </button>
            )}
            {profile.pca_data && (
            <button
              onClick={() => setActiveTab('pca')}
              className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                activeTab === 'pca'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <ScatterIcon className="w-4 h-4" />
              Multivariate (PCA)
            </button>
            )}
            {profile.geospatial && (
                <button
                onClick={() => setActiveTab('geospatial')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                    activeTab === 'geospatial'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                >
                <Map className="w-4 h-4" />
                Geospatial
                </button>
            )}
            {profile.timeseries && (
                <button
                onClick={() => setActiveTab('timeseries')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                    activeTab === 'timeseries'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                >
                <Calendar className="w-4 h-4" />
                Time Series
                </button>
            )}
            {profile.correlations && profile.correlations.values && profile.correlations.values.length > 0 && (
            <button
              onClick={() => setActiveTab('correlations')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'correlations'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Correlations
            </button>
            )}
            {profile.target_col && profile.target_correlations && Object.keys(profile.target_correlations).length > 0 && (
                <button
                onClick={() => setActiveTab('target')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === 'target'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                >
                Target Analysis
                </button>
            )}
          </nav>
        </div>

        {/* Tab Content */}
        {activeTab === 'insights' && (
            <div className="mt-4 space-y-6">
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                        <Lightbulb className="w-5 h-5 mr-2 text-yellow-500" />
                        Actionable Recommendations
                    </h3>
                    {profile.recommendations && profile.recommendations.length > 0 ? (
                        <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                            {profile.recommendations.map((rec: any, idx: number) => (
                                <div key={idx} className="flex items-start p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-100 dark:border-gray-800">
                                    <div className={`p-2 rounded-full mr-4 shrink-0 ${
                                        rec.action === 'Drop' ? 'bg-red-100 text-red-600' :
                                        rec.action === 'Impute' ? 'bg-blue-100 text-blue-600' :
                                        rec.action === 'Transform' ? 'bg-purple-100 text-purple-600' :
                                        rec.action === 'Resample' ? 'bg-orange-100 text-orange-600' :
                                        'bg-green-100 text-green-600'
                                    }`}>
                                        {rec.action === 'Drop' ? <X className="w-4 h-4" /> : 
                                            rec.action === 'Keep' ? <CheckCircle className="w-4 h-4" /> :
                                            <RefreshCw className="w-4 h-4" />}
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
                                            {rec.action} {rec.column && <span className="text-gray-500">'{rec.column}'</span>}
                                        </h4>
                                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{rec.suggestion}</p>
                                        <p className="text-xs text-gray-400 mt-2 italic">Reason: {rec.reason}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-gray-500 italic">No specific recommendations found. Your data looks clean!</p>
                    )}
                </div>
            </div>
        )}

        {activeTab === 'pca' && (
            <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                        <ScatterIcon className="w-5 h-5 mr-2 text-purple-500" />
                        Multivariate Structure (PCA)
                        <InfoTooltip text="Reduces data to 2D or 3D. Points closer together are similar. Colors show clusters or target values." />
                    </h3>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => downloadChart('pca-chart', 'pca-analysis', 'Multivariate Structure (PCA)', isPCA3D ? '3D Projection' : '2D Projection')}
                            className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                            title="Download Chart"
                        >
                            <Download className="w-4 h-4" />
                        </button>
                        <button
                            onClick={() => setIsPCA3D(!isPCA3D)}
                            className={`p-2 rounded-md border transition-colors flex items-center gap-2 text-sm ${
                                isPCA3D 
                                    ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300' 
                                    : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300'
                            }`}
                        >
                            <Box className="w-4 h-4" />
                            {isPCA3D ? 'Switch to 2D' : 'Switch to 3D'}
                        </button>
                    </div>
                </div>

                <div id="pca-chart">
                {profile.pca_data ? (
                    isPCA3D ? (
                        <ThreeDScatterPlot 
                            data={profile.pca_data} 
                            xKey="x" 
                            yKey="y" 
                            zKey="z"
                            labelKey="label"
                            xLabel="PC1"
                            yLabel="PC2"
                            zLabel="PC3"
                        />
                    ) : (
                        <CanvasScatterPlot 
                            data={profile.pca_data} 
                            xKey="x" 
                            yKey="y" 
                            labelKey="label"
                            xLabel="Principal Component 1 (PC1)"
                            yLabel="Principal Component 2 (PC2)"
                        />
                    )
                ) : (
                    <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded text-gray-400 text-sm text-center p-4">
                        Not enough numeric data for PCA.
                    </div>
                )}
                </div>

                <div className="mt-6 flex items-start gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md border border-blue-100 dark:border-blue-800">
                    <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
                    <div className="text-sm text-blue-700 dark:text-blue-300">
                        <p className="font-medium mb-1">About this visualization</p>
                        <p>
                            This chart shows a {isPCA3D ? '3D' : '2D'} projection of your data using Principal Component Analysis (PCA). 
                            Points that are close together share similar characteristics across all numeric features.
                        </p>
                        <p className="mt-1 text-xs opacity-80">
                            * For performance, this visualization uses a random sample of up to <strong>5,000 points</strong>.
                        </p>
                    </div>
                </div>
            </div>
        )}

        {activeTab === 'geospatial' && profile.geospatial && (
            <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                        <Map className="w-5 h-5 mr-2 text-blue-500" />
                        Geospatial Analysis
                        <InfoTooltip text="Visualizes data points on a map based on detected Latitude/Longitude columns. Colors indicate target values." />
                    </h2>
                    <div className="text-sm text-gray-500">
                        Detected columns: <span className="font-mono bg-gray-100 dark:bg-gray-900 px-1 rounded">{profile.geospatial.lat_col}</span>, <span className="font-mono bg-gray-100 dark:bg-gray-900 px-1 rounded">{profile.geospatial.lon_col}</span>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 h-96 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 relative z-0">
                        {(() => {
                            // Calculate bounds for the map
                            const bounds: [[number, number], [number, number]] = [
                                [profile.geospatial.min_lat, profile.geospatial.min_lon],
                                [profile.geospatial.max_lat, profile.geospatial.max_lon]
                            ];
                            
                            // Helper to get color
                            const uniqueLabels = Array.from(new Set(profile.geospatial.sample_points.map((p: any) => p.label))).filter(Boolean);
                            const isChaotic = uniqueLabels.length > 20;
                            
                            const getColor = (label: string | null) => {
                                if (!label) return '#3b82f6'; // Default blue
                                if (isChaotic) return '#3b82f6'; // Single color for chaotic
                                const index = uniqueLabels.indexOf(label);
                                return COLORS[index % COLORS.length];
                            };

                            return (
                                <MapContainer 
                                    bounds={bounds} 
                                    style={{ height: '100%', width: '100%' }}
                                    scrollWheelZoom={true}
                                >
                                    <TileLayer
                                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                    />
                                    {profile.geospatial.sample_points.map((point: any, idx: number) => (
                                        <CircleMarker 
                                            key={idx} 
                                            center={[point.lat, point.lon]} 
                                            radius={5}
                                            pathOptions={{ 
                                                color: getColor(point.label), 
                                                fillColor: getColor(point.label), 
                                                fillOpacity: 0.7,
                                                weight: 1
                                            }}
                                        >
                                            <Popup>
                                                <div className="text-xs">
                                                    <strong>Lat:</strong> {point.lat.toFixed(4)}<br/>
                                                    <strong>Lon:</strong> {point.lon.toFixed(4)}<br/>
                                                    {point.label && <><strong>{profile.target_col}:</strong> {point.label}</>}
                                                </div>
                                            </Popup>
                                        </CircleMarker>
                                    ))}
                                </MapContainer>
                            );
                        })()}
                    </div>

                    <div className="space-y-4">
                        <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                            <h3 className="text-sm font-medium text-gray-500 mb-3">Spatial Bounds</h3>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="block text-xs text-gray-400">Latitude Range</span>
                                    <span className="font-mono">{profile.geospatial.min_lat.toFixed(4)} to {profile.geospatial.max_lat.toFixed(4)}</span>
                                </div>
                                <div>
                                    <span className="block text-xs text-gray-400">Longitude Range</span>
                                    <span className="font-mono">{profile.geospatial.min_lon.toFixed(4)} to {profile.geospatial.max_lon.toFixed(4)}</span>
                                </div>
                            </div>
                        </div>

                        <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                            <h3 className="text-sm font-medium text-gray-500 mb-3">Centroid</h3>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-red-500"></div>
                                <span className="font-mono text-sm">
                                    {profile.geospatial.centroid_lat.toFixed(4)}, {profile.geospatial.centroid_lon.toFixed(4)}
                                </span>
                            </div>
                        </div>
                        
                        <div className="flex items-start gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md border border-blue-100 dark:border-blue-800">
                            <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
                            <div className="text-sm text-blue-700 dark:text-blue-300">
                                <p>
                                    This scatter plot shows the spatial distribution of your data.
                                    {profile.target_col && " Points are colored by the target variable."}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        )}

        {activeTab === 'target' && profile.target_col && profile.target_correlations && (
            <div className="mt-4 grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-3 space-y-6">
                    {/* Main Analysis Content */}
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                            <BarChart2 className="w-5 h-5 mr-2 text-blue-500" />
                            Target Analysis: <span className="ml-2 text-blue-600">{profile.target_col}</span>
                            <InfoTooltip text="Analyzes relationships between the target variable and other features. Shows top correlations and key drivers." />
                        </h2>
                        <button
                            onClick={() => downloadChart('target-analysis-chart', 'target-analysis', `Target Analysis: ${profile.target_col}`, 'Top Associated Features')}
                            className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                            title="Download Chart"
                        >
                            <Download className="w-4 h-4" />
                        </button>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-2">
                            <h3 className="text-sm font-medium text-gray-500 mb-4">Top Associated Features</h3>
                            <div className="h-64 w-full" id="target-analysis-chart">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart 
                                        data={Object.entries(profile.target_correlations)
                                            .slice(0, 10)
                                            .map(([k, v]) => ({ name: k, value: v as number }))
                                        } 
                                        layout="vertical"
                                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                        <XAxis type="number" domain={[0, 1]} />
                                        <YAxis type="category" dataKey="name" width={120} tick={{fontSize: 12}} />
                                        <Tooltip 
                                            contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                            itemStyle={{ color: '#fff' }}
                                            formatter={(value: number) => [value.toFixed(3), 'Association']}
                                        />
                                        <ReferenceLine x={0} stroke="#9ca3af" />
                                        <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                            <h3 className="text-sm font-medium text-gray-500 mb-2">Key Insights</h3>
                            <ul className="space-y-2 text-sm">
                                <li className="flex items-start">
                                    <span className="mr-2 text-blue-500">•</span>
                                    <span>Target is <strong>{profile.columns[profile.target_col]?.dtype}</strong></span>
                                </li>
                                <li className="flex items-start">
                                    <span className="mr-2 text-blue-500">•</span>
                                    <span>
                                        Strongest predictor: <strong>{Object.keys(profile.target_correlations)[0] || 'None'}</strong> 
                                        {Object.values(profile.target_correlations)[0] !== undefined && (
                                            <> ({(Object.values(profile.target_correlations)[0] as number).toFixed(2)})</>
                                        )}
                                    </span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Detailed Interactions (Box Plots) */}
                {profile.target_interactions && profile.target_interactions.length > 0 && (
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                        <div className="flex justify-between items-center mb-2">
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                                <ScatterIcon className="w-5 h-5 mr-2 text-purple-500" />
                                Detailed Interactions (Box Plots)
                                <InfoTooltip text="Shows distribution of values across categories. The box represents the middle 50% of data (Q1 to Q3). The line inside is the median." />
                            </h3>
                            <button
                                onClick={downloadAllInteractions}
                                className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                                title="Download All Charts"
                            >
                                <Download className="w-4 h-4" />
                            </button>
                        </div>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                            <span className="font-semibold">ANOVA Analysis:</span> A p-value &lt; 0.05 (highlighted in green) indicates that the feature varies significantly across the target categories, making it a strong predictor.
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {profile.target_interactions.map((interaction: any, idx: number) => (
                                <div key={idx} className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800 relative group">
                                    <div className="absolute top-2 right-2 opacity-100">
                                        <button
                                            onClick={() => downloadChart(
                                                `interaction-chart-${idx}`, 
                                                `interaction-${interaction.feature}`,
                                                `${interaction.feature} vs ${profile.target_col}`,
                                                interaction.p_value !== undefined ? `ANOVA p: ${Number(interaction.p_value).toExponential(2)}` : undefined
                                            )}
                                            className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                                            title="Download Chart"
                                        >
                                            <Download className="w-3 h-3" />
                                        </button>
                                    </div>
                                    <div className="text-center mb-2">
                                        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                            {interaction.feature} vs {profile.target_col}
                                        </h4>
                                        {interaction.p_value !== undefined && interaction.p_value !== null && (
                                            <span className={`text-xs px-2 py-0.5 rounded-full inline-block mt-1 ${
                                                interaction.p_value < 0.05 
                                                    ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' 
                                                    : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                                            }`}>
                                                ANOVA p: {Number(interaction.p_value).toExponential(2)}
                                            </span>
                                        )}
                                    </div>
                                    <div className="h-64 w-full" id={`interaction-chart-${idx}`}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <ComposedChart 
                                                data={interaction.data.map((d: any) => ({
                                                    ...d,
                                                    boxBottom: d.stats.q1,
                                                    boxHeight: d.stats.q3 - d.stats.q1
                                                }))} 
                                                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                            >
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                                                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
                                                <Tooltip 
                                                    cursor={{ fill: 'transparent' }}
                                                    content={({ active, payload }) => {
                                                        if (active && payload && payload.length) {
                                                            const data = payload[0].payload;
                                                            return (
                                                                <div className="bg-gray-800 text-white text-xs p-2 rounded shadow-lg z-50">
                                                                    <p className="font-semibold mb-1">{data.name}</p>
                                                                    <p>Max: {data.stats.max.toFixed(2)}</p>
                                                                    <p>Q3: {data.stats.q3.toFixed(2)}</p>
                                                                    <p>Median: {data.stats.median.toFixed(2)}</p>
                                                                    <p>Q1: {data.stats.q1.toFixed(2)}</p>
                                                                    <p>Min: {data.stats.min.toFixed(2)}</p>
                                                                </div>
                                                            );
                                                        }
                                                        return null;
                                                    }}
                                                />
                                                {/* Invisible bar to push the box up to Q1 */}
                                                <Bar dataKey="boxBottom" stackId="a" fill="transparent" legendType="none" />
                                                {/* The Box (Q1 to Q3) */}
                                                <Bar dataKey="boxHeight" stackId="a" fill="#8884d8" fillOpacity={0.6} name="IQR (Q1-Q3)" />
                                                
                                                {/* Median, Min, Max Points */}
                                                <Scatter name="Median" dataKey="stats.median" fill="#ff7300" shape="square" />
                                                <Scatter name="Max" dataKey="stats.max" fill="#82ca9d" shape="cross" />
                                                <Scatter name="Min" dataKey="stats.min" fill="#82ca9d" shape="cross" />
                                            </ComposedChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <p className="text-xs text-center text-gray-400 mt-2">
                                        * Box: Q1-Q3. Orange square: Median. Green crosses: Min/Max.
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                </div>

                {/* History Sidebar */}
                <div className="lg:col-span-1">
                    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm h-fit sticky top-4">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center justify-between">
                        <div className="flex items-center">
                            <Clock className="w-4 h-4 mr-2" />
                            Analysis History
                        </div>
                        {loading && <Loader2 className="w-3 h-3 animate-spin text-blue-500" />}
                    </h3>
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                        {history.length === 0 ? (
                            <p className="text-xs text-gray-500 italic">No previous analyses found.</p>
                        ) : (
                            history.slice(0, 10).map((item) => (
                                <button
                                    key={item.id}
                                    onClick={() => !loading && loadSpecificReport(item.id)}
                                    disabled={loading}
                                    className={`w-full text-left p-3 rounded-md text-sm border transition-colors ${
                                        report && report.id === item.id
                                            ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300'
                                            : 'bg-gray-50 border-gray-100 text-gray-600 hover:bg-gray-100 dark:bg-gray-900 dark:border-gray-800 dark:text-gray-400 dark:hover:bg-gray-800'
                                    } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                >
                                    <div className="font-medium truncate">
                                        {item.target_col ? `Target: ${item.target_col}` : 'General Analysis'}
                                    </div>
                                    <div className="text-xs opacity-70 mt-1">
                                        {new Date(item.created_at).toLocaleString()}
                                    </div>
                                    <div className={`text-xs mt-1 inline-block px-1.5 py-0.5 rounded ${
                                        item.status === 'COMPLETED' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                                        item.status === 'FAILED' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                                    }`}>
                                        {item.status}
                                    </div>
                                </button>
                            ))
                        )}
                    </div>
                </div>
                </div>
            </div>
        )}

        {/* Tab Content */}
        {activeTab === 'timeseries' && profile.timeseries && (
            <div className="mt-4 space-y-6">
                {/* Trend Chart */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center justify-between">
                        <div className="flex items-center">
                            <Calendar className="w-5 h-5 mr-2 text-blue-500" />
                            Trend Analysis ({profile.timeseries.date_col})
                            <InfoTooltip text="Shows how values change over time. Look for long-term trends (up/down) or sudden shifts." />
                        </div>
                        <div className="flex items-center gap-2">
                            {profile.timeseries.stationarity_test && (
                                <div className={`text-xs px-3 py-1 rounded-full border flex items-center gap-1 ${
                                    profile.timeseries.stationarity_test.is_stationary 
                                        ? 'bg-green-50 border-green-200 text-green-700 dark:bg-green-900/20 dark:border-green-800 dark:text-green-300' 
                                        : 'bg-yellow-50 border-yellow-200 text-yellow-700 dark:bg-yellow-900/20 dark:border-yellow-800 dark:text-yellow-300'
                                }`}>
                                    <InfoTooltip text="Augmented Dickey-Fuller Test. Stationary (p<0.05) means the data has constant mean/variance over time. Non-Stationary (p>=0.05) means it has a trend or seasonality." />
                                    <span className="font-semibold">ADF Test:</span> {profile.timeseries.stationarity_test.is_stationary ? 'Stationary' : 'Non-Stationary'} 
                                    <span className="opacity-75 ml-1">(p={profile.timeseries.stationarity_test.p_value.toFixed(3)})</span>
                                </div>
                            )}
                            <button
                                onClick={() => downloadChart('trend-chart', 'trend-analysis', `Trend Analysis (${profile.timeseries.date_col})`, profile.timeseries.stationarity_test ? `ADF Test: ${profile.timeseries.stationarity_test.is_stationary ? 'Stationary' : 'Non-Stationary'} (p=${profile.timeseries.stationarity_test.p_value.toFixed(3)})` : undefined)}
                                className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                                title="Download Chart"
                            >
                                <Download className="w-4 h-4" />
                            </button>
                        </div>
                    </h3>
                    <div className="h-80 w-full" id="trend-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={profile.timeseries.trend}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis 
                                    dataKey="date" 
                                    tick={{ fontSize: 12 }} 
                                    minTickGap={30}
                                />
                                <YAxis />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                    labelStyle={{ color: '#9ca3af' }}
                                />
                                <Legend />
                                {Object.keys(profile.timeseries.trend[0]?.values || {}).map((key, idx) => (
                                    <Line 
                                        key={key}
                                        type="monotone" 
                                        dataKey={`values.${key}`} 
                                        name={key}
                                        stroke={COLORS[idx % COLORS.length]} 
                                        dot={false}
                                        strokeWidth={2}
                                    />
                                ))}
                                {/* Fallback if no values (just count) */}
                                {(!profile.timeseries.trend[0]?.values || Object.keys(profile.timeseries.trend[0].values).length === 0) && (
                                     <Line type="monotone" dataKey="count" stroke="#3b82f6" dot={false} />
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Seasonality Grid */}
                <div className="grid grid-cols-1 gap-6">
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm relative group">
                        <div className="absolute top-4 right-4 opacity-100 z-10">
                            <button
                                onClick={() => downloadChart('day-seasonality-chart', 'day-seasonality', 'Day of Week Seasonality')}
                                className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                                title="Download Chart"
                            >
                                <Download className="w-3 h-3" />
                            </button>
                        </div>
                        <div className="flex items-center mb-4">
                            <h3 className="text-sm font-medium text-gray-500">Day of Week Seasonality</h3>
                            <InfoTooltip text="Average values for each day. Helps identify weekly patterns (e.g., lower on weekends)." />
                        </div>
                        <div className="h-64 w-full" id="day-seasonality-chart">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={profile.timeseries.seasonality.day_of_week}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                    <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                                    <YAxis />
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                        cursor={{fill: 'transparent'}}
                                    />
                                    <Bar dataKey="count" fill="#8884d8" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* Monthly Seasonality (Full Width) */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm relative group">
                    <div className="absolute top-4 right-4 opacity-100 z-10">
                        <button
                            onClick={() => downloadChart('month-seasonality-chart', 'month-seasonality', 'Monthly Seasonality')}
                            className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                            title="Download Chart"
                        >
                            <Download className="w-3 h-3" />
                        </button>
                    </div>
                    <div className="flex items-center mb-4">
                        <h3 className="text-sm font-medium text-gray-500">Monthly Seasonality</h3>
                        <InfoTooltip text="Average values for each month. Helps identify yearly patterns or seasonal effects." />
                    </div>
                    <div className="h-64 w-full" id="month-seasonality-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={profile.timeseries.seasonality.month_of_year}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                                <YAxis />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                    cursor={{fill: 'transparent'}}
                                />
                                <Bar dataKey="count" fill="#82ca9d" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Autocorrelation (ACF) */}
                {profile.timeseries.autocorrelation && profile.timeseries.autocorrelation.length > 0 && (
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm relative group">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center">
                                <h3 className="text-sm font-medium text-gray-500">Autocorrelation (Lag Analysis)</h3>
                                <div className="group relative ml-2">
                                    <Info className="w-4 h-4 text-gray-400 cursor-help" />
                                    <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none text-center">
                                        Shows how much the current value depends on past values (lags). 
                                        High bars indicate repeating patterns or seasonality.
                                        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-800"></div>
                                    </div>
                                </div>
                            </div>
                            <button
                                onClick={() => downloadChart('autocorrelation-chart', 'autocorrelation', 'Autocorrelation (Lag Analysis)')}
                                className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                                title="Download Chart"
                            >
                                <Download className="w-3 h-3" />
                            </button>
                        </div>
                        <div className="h-80 w-full pb-6" id="autocorrelation-chart">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={profile.timeseries.autocorrelation} margin={{ bottom: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                    <XAxis 
                                        dataKey="lag" 
                                        tick={{ fontSize: 12 }} 
                                        label={{ value: 'Lag (Days)', position: 'insideBottom', offset: -15, fontSize: 12 }}
                                    />
                                    <YAxis domain={[-1, 1]} />
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                        cursor={{fill: 'transparent'}}
                                        formatter={(value: number) => [value.toFixed(3), 'Correlation']}
                                    />
                                    <ReferenceLine y={0} stroke="#9ca3af" />
                                    <Bar dataKey="corr" fill="#f59e0b" radius={[2, 2, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>
        )}

        {activeTab === 'variables' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {/* Active Columns */}
            {Object.values(profile.columns).map((col: any) => (
              <VariableCard 
                key={col.name} 
                profile={col} 
                onClick={() => setSelectedVariable(col)} 
                onToggleExclude={handleToggleExclude}
                isExcluded={false}
              />
            ))}
            
            {/* Excluded Columns (Ghost Cards) */}
            {profile.excluded_columns && profile.excluded_columns.map((colName: string) => (
                <VariableCard 
                    key={colName}
                    profile={{ name: colName, dtype: 'Excluded', missing_percentage: 0 }}
                    onClick={() => {}}
                    onToggleExclude={handleToggleExclude}
                    isExcluded={true}
                />
            ))}
          </div>
        )}

        {activeTab === 'bivariate' && (
          <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                    <ScatterIcon className="w-5 h-5 mr-2 text-blue-500" />
                    Bivariate Analysis (Scatter Plot)
                    <InfoTooltip text="Visualize the relationship between two numeric variables." />
                </h3>
                <button
                    onClick={() => downloadChart('bivariate-chart', 'bivariate-analysis', 'Bivariate Analysis', `${scatterX} vs ${scatterY}`)}
                    className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                    title="Download Chart"
                >
                    <Download className="w-4 h-4" />
                </button>
            </div>
            
            <div className="flex gap-4 mb-6 items-end">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">X Axis</label>
                    <select 
                        value={scatterX} 
                        onChange={(e) => setScatterX(e.target.value)}
                        className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                    >
                        <option value="">Select Variable</option>
                        {Object.values(profile.columns)
                            .filter((c: any) => c.dtype === 'Numeric')
                            .map((c: any) => (
                                <option key={c.name} value={c.name}>{c.name}</option>
                            ))
                        }
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Y Axis</label>
                    <select 
                        value={scatterY} 
                        onChange={(e) => setScatterY(e.target.value)}
                        className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                    >
                        <option value="">Select Variable</option>
                        {Object.values(profile.columns)
                            .filter((c: any) => c.dtype === 'Numeric')
                            .map((c: any) => (
                                <option key={c.name} value={c.name}>{c.name}</option>
                            ))
                        }
                    </select>
                </div>
                
                {is3D && (
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Z Axis</label>
                        <select 
                            value={scatterZ} 
                            onChange={(e) => setScatterZ(e.target.value)}
                            className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                        >
                            <option value="">Select Variable</option>
                            {Object.values(profile.columns)
                                .filter((c: any) => c.dtype === 'Numeric')
                                .map((c: any) => (
                                    <option key={c.name} value={c.name}>{c.name}</option>
                                ))
                            }
                        </select>
                    </div>
                )}

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Color By</label>
                    <select 
                        value={scatterColor} 
                        onChange={(e) => setScatterColor(e.target.value)}
                        className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                    >
                        <option value="">None</option>
                        {Object.values(profile.columns)
                            .map((c: any) => (
                                <option key={c.name} value={c.name}>{c.name}</option>
                            ))
                        }
                    </select>
                </div>

                <button
                    onClick={() => setIs3D(!is3D)}
                    className={`p-2 rounded-md border transition-colors ${
                        is3D 
                            ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300' 
                            : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300'
                    }`}
                    title="Toggle 3D View"
                >
                    <Box className="w-5 h-5" />
                </button>
            </div>

            <div id="bivariate-chart">
            {scatterX && scatterY ? (
                is3D && scatterZ ? (
                    <ThreeDScatterPlot 
                        data={profile.sample_data} 
                        xKey={scatterX} 
                        yKey={scatterY} 
                        zKey={scatterZ}
                        labelKey={scatterColor || undefined}
                        xLabel={scatterX}
                        yLabel={scatterY}
                        zLabel={scatterZ}
                    />
                ) : (
                    <CanvasScatterPlot 
                        data={profile.sample_data} 
                        xKey={scatterX} 
                        yKey={scatterY} 
                        labelKey={scatterColor || undefined}
                        xLabel={scatterX}
                        yLabel={scatterY}
                    />
                )
            ) : (
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded text-gray-400 text-sm">
                    Select X and Y variables to generate scatter plot.
                </div>
            )}
            </div>
          </div>
        )}

        {activeTab === 'outliers' && profile.outliers && (
            <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                    <AlertTriangle className="w-5 h-5 mr-2 text-red-500" />
                    Outlier Analysis ({profile.outliers.method})
                    <InfoTooltip text="Detects anomalous rows using Isolation Forest. Lower scores indicate higher anomaly." />
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-100 dark:border-red-800">
                        <span className="text-sm text-red-600 dark:text-red-400 block mb-1">Total Outliers</span>
                        <span className="text-2xl font-bold text-red-700 dark:text-red-300">{profile.outliers.total_outliers}</span>
                    </div>
                    <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-100 dark:border-red-800">
                        <span className="text-sm text-red-600 dark:text-red-400 block mb-1">Percentage</span>
                        <span className="text-2xl font-bold text-red-700 dark:text-red-300">{profile.outliers.outlier_percentage.toFixed(2)}%</span>
                    </div>
                </div>

                <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">Top Anomalous Rows</h4>
                <div className="overflow-x-auto border border-gray-200 dark:border-gray-700 rounded-lg">
                    <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead className="bg-gray-50 dark:bg-gray-900">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Row Index</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Anomaly Score</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Why is this an outlier?</th>
                                {profile.outliers.top_outliers[0] && Object.keys(profile.outliers.top_outliers[0].values).slice(0, 5).map(key => (
                                    <th key={key} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{key}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                            {profile.outliers.top_outliers.map((outlier: any) => (
                                <tr key={outlier.index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                                    <td className="px-4 py-3 text-sm font-mono text-gray-500">{outlier.index}</td>
                                    <td className="px-4 py-3 text-sm font-mono text-red-600">{outlier.score.toFixed(4)}</td>
                                    <td className="px-4 py-3 text-sm">
                                        {outlier.explanation ? (
                                            <div className="space-y-1">
                                                {outlier.explanation.map((exp: any, i: number) => (
                                                    <div key={i} className="text-xs">
                                                        <span className="font-semibold text-gray-700 dark:text-gray-300">{exp.feature}:</span>{' '}
                                                        <span className="text-red-600 dark:text-red-400">{exp.value.toFixed(2)}</span>{' '}
                                                        <span className="text-gray-400">
                                                            (Median: {exp.median.toFixed(2)}, Diff: {exp.diff_pct.toFixed(0)}%)
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <span className="text-gray-400 text-xs">No explanation available</span>
                                        )}
                                    </td>
                                    {Object.entries(outlier.values).slice(0, 5).map(([key, val]: any) => (
                                        <td key={key} className="px-4 py-3 text-sm text-gray-900 dark:text-gray-300">
                                            {typeof val === 'number' ? val.toFixed(2) : String(val)}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        )}

        {activeTab === 'correlations' && profile.correlations && (
          <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                    <BarChart2 className="w-5 h-5 mr-2 text-blue-500" />
                    Correlation Matrix
                    <InfoTooltip text="Shows linear relationships between variables. 1 is perfect positive, -1 is perfect negative correlation." />
                </h3>
                <button
                    onClick={downloadCorrelationHeatmap}
                    className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                    title="Download Chart"
                >
                    <Download className="w-4 h-4" />
                </button>
            </div>
            <div id="correlation-heatmap">
                <CorrelationHeatmap data={profile.correlations} />
            </div>
          </div>
        )}

        {activeTab === 'sample' && profile.sample_data && (
          <div className="mt-4 overflow-x-auto bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  {Object.keys(profile.sample_data[0] || {}).map((col) => (
                    <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider group">
                      <div className="flex items-center gap-2">
                        {col}
                        <button 
                            onClick={(e) => {
                                e.stopPropagation();
                                handleToggleExclude(col, !excludedCols.includes(col));
                            }}
                            className={`p-1 rounded transition-colors opacity-0 group-hover:opacity-100 ${
                                excludedCols.includes(col)
                                    ? 'hover:bg-green-100 text-gray-400 hover:text-green-600 dark:hover:bg-green-900/30 opacity-100' 
                                    : 'hover:bg-red-100 text-gray-400 hover:text-red-500 dark:hover:bg-red-900/30'
                            }`}
                            title={excludedCols.includes(col) ? "Include in analysis" : "Exclude from analysis"}
                        >
                            {excludedCols.includes(col) ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                        </button>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {profile.sample_data.slice(0, 50).map((row: any, i: number) => (
                  <tr key={i}>
                    {Object.entries(row).map(([key, val]: [string, any], j: number) => (
                      <td key={j} className={`px-6 py-4 whitespace-nowrap text-sm ${excludedCols.includes(key) ? 'text-gray-300 dark:text-gray-600 line-through' : 'text-gray-500 dark:text-gray-400'}`}>
                        {val !== null ? String(val) : <span className="italic text-gray-400">null</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {profile.sample_data.length > 50 && (
                <div className="p-4 text-center text-sm text-gray-500 border-t border-gray-200 dark:border-gray-700">
                    Showing first 50 rows of {profile.sample_data.length} sample rows.
                </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div id="eda-report-container" className="p-6 max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-4">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Exploratory Data Analysis</h1>
                {report && report.created_at && (
                    <p className="text-xs text-gray-500 mt-1">
                        Last analyzed: {new Date(report.created_at).toLocaleString()}
                    </p>
                )}
            </div>
            <div className="flex gap-2 no-print">
                <button 
                    onClick={() => setShowHistoryModal(true)}
                    className="flex items-center px-3 py-1.5 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors ml-4"
                >
                    <List className="w-4 h-4 mr-2" />
                    Jobs History
                </button>
            </div>
        </div>
        <div className="flex items-center gap-4 no-print">
          <select
            value={selectedDataset || ''}
            onChange={(e) => setSelectedDataset(Number(e.target.value))}
            className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
          >
            <option value="" disabled>Select a dataset</option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>{ds.name}</option>
            ))}
          </select>

          <div className="flex items-center gap-2">
             <span className="text-sm text-gray-500">Target:</span>
             <select
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                disabled={!report || !report.profile_data}
                className="block w-40 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border disabled:opacity-50"
             >
                <option value="">None</option>
                {report && report.profile_data && Object.keys(report.profile_data.columns).map(col => (
                    <option key={col} value={col}>{col}</option>
                ))}
             </select>
          </div>

          <button 
            onClick={() => selectedDataset && runAnalysis()}
            disabled={!selectedDataset || analyzing}
            className="p-2 text-gray-500 hover:text-blue-600 rounded-full hover:bg-gray-100"
            title="Refresh Analysis"
          >
            <RefreshCw className={`w-5 h-5 ${analyzing ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Active Filters Bar */}
      <div className="mb-6 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md border border-blue-100 dark:border-blue-800">
          <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center mr-2">
                  <Filter className="w-4 h-4 text-blue-600 dark:text-blue-400 mr-2" />
                  <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Active Filters</span>
                  <InfoTooltip text="Filters are applied to the raw dataset before profiling. You can add filters manually or by clicking on distribution bars to filter on specific range." />
              </div>
              
              {filters.map((filter, idx) => (
                  <div key={idx} className="flex items-center bg-white dark:bg-gray-800 border border-blue-200 dark:border-blue-700 rounded-full px-3 py-1 text-sm shadow-sm animate-in fade-in zoom-in duration-200">
                      <span className="font-medium text-gray-700 dark:text-gray-300 mr-1">{filter.column}</span>
                      <span className="text-gray-500 mx-1">{filter.operator}</span>
                      <span className="font-mono text-blue-600 dark:text-blue-400">{String(filter.value)}</span>
                      <button 
                          onClick={() => handleRemoveFilter(idx)}
                          className="ml-2 text-gray-400 hover:text-red-500"
                      >
                          <X className="w-3 h-3" />
                      </button>
                  </div>
              ))}

              {!showFilterForm ? (
                  <button 
                      onClick={() => setShowFilterForm(true)}
                      className="flex items-center px-2 py-1 text-xs font-medium text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 bg-white dark:bg-gray-800 border border-blue-200 dark:border-blue-700 rounded-full hover:bg-blue-50 dark:hover:bg-gray-700 transition-colors"
                  >
                      <Plus className="w-3 h-3 mr-1" />
                      Add Filter
                  </button>
              ) : (
                  <div className="flex items-center gap-2 bg-white dark:bg-gray-800 p-1 rounded-md border border-blue-200 dark:border-blue-700 animate-in slide-in-from-left-2 duration-200">
                      <select 
                          value={newFilterCol}
                          onChange={(e) => setNewFilterCol(e.target.value)}
                          className="text-xs border-none bg-transparent focus:ring-0 py-1 pl-2 pr-6"
                      >
                          <option value="" disabled>Column</option>
                          {report && report.profile_data && Object.keys(report.profile_data.columns).map(col => (
                              <option key={col} value={col}>{col}</option>
                          ))}
                      </select>
                      <select 
                          value={newFilterOp}
                          onChange={(e) => setNewFilterOp(e.target.value)}
                          className="text-xs border-none bg-gray-50 dark:bg-gray-900 rounded focus:ring-0 py-1 px-2"
                      >
                          <option value="==">==</option>
                          <option value="!=">!=</option>
                          <option value=">">&gt;</option>
                          <option value="<">&lt;</option>
                          <option value=">=">&gt;=</option>
                          <option value="<=">&lt;=</option>
                      </select>
                      <input 
                          type="text" 
                          value={newFilterVal}
                          onChange={(e) => setNewFilterVal(e.target.value)}
                          placeholder="Value"
                          className="text-xs border-none bg-transparent focus:ring-0 py-1 w-20"
                          onKeyDown={(e) => {
                              if (e.key === 'Enter' && newFilterCol && newFilterVal) {
                                  handleAddFilter(newFilterCol, isNaN(Number(newFilterVal)) ? newFilterVal : Number(newFilterVal), newFilterOp);
                                  setShowFilterForm(false);
                                  setNewFilterCol('');
                                  setNewFilterVal('');
                              }
                          }}
                      />
                      <button 
                          onClick={() => {
                              if (newFilterCol && newFilterVal) {
                                  handleAddFilter(newFilterCol, isNaN(Number(newFilterVal)) ? newFilterVal : Number(newFilterVal), newFilterOp);
                                  setShowFilterForm(false);
                                  setNewFilterCol('');
                                  setNewFilterVal('');
                              }
                          }}
                          disabled={!newFilterCol || !newFilterVal}
                          className="p-1 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                      >
                          <Plus className="w-3 h-3" />
                      </button>
                      <button 
                          onClick={() => setShowFilterForm(false)}
                          className="p-1 text-gray-400 hover:text-gray-600"
                      >
                          <X className="w-3 h-3" />
                      </button>
                  </div>
              )}

              {filters.length > 0 && (
                  <button 
                      onClick={() => { setFilters([]); runAnalysis(undefined, []); }}
                      className="ml-auto text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 underline"
                  >
                      Clear All
                  </button>
              )}
          </div>
      </div>

      {renderContent()}

      {/* Variable Detail Modal */}
      {selectedVariable && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto flex flex-col animate-in fade-in zoom-in duration-200">
            <div className="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700 sticky top-0 bg-white dark:bg-gray-800 z-10">
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                  {selectedVariable.name}
                  <span className={`text-sm font-normal px-2 py-0.5 rounded-full ${
                    selectedVariable.dtype === 'Numeric' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                    selectedVariable.dtype === 'Categorical' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300' :
                    'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                  }`}>
                    {selectedVariable.dtype}
                  </span>
                </h2>
              </div>
              <button 
                onClick={() => setSelectedVariable(null)}
                className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            
            <div className="p-6 space-y-6">
              {/* Distribution Chart */}
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800">
                <div className="flex justify-between items-center mb-4">
                    <div className="flex items-center">
                        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mr-2">Distribution</h3>
                        <InfoTooltip text="Click on any bar to filter the entire dataset by that range or category." />
                    </div>
                    <button
                        onClick={() => downloadChart('distribution-chart', `distribution-${selectedVariable.name}`, `Distribution: ${selectedVariable.name}`)}
                        className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                        title="Download Chart"
                    >
                        <Download className="w-3 h-3" />
                    </button>
                </div>
                <div id="distribution-chart">
                    <DistributionChart 
                        profile={selectedVariable} 
                        onBarClick={(data) => {
                            if ((selectedVariable.dtype === 'Numeric' || selectedVariable.dtype === 'DateTime') && data.rawBin) {
                                // Add range filter (>= start AND < end)
                                const newFilters = [
                                    ...filters,
                                    { column: selectedVariable.name, operator: '>=', value: data.rawBin.start },
                                    { column: selectedVariable.name, operator: '<', value: data.rawBin.end }
                                ];
                                setFilters(newFilters);
                                runAnalysis(undefined, newFilters);
                                setSelectedVariable(null);
                            } else if (selectedVariable.dtype === 'Categorical') {
                                handleAddFilter(selectedVariable.name, data.value, '==');
                                setSelectedVariable(null);
                            }
                        }}
                    />
                </div>
              </div>

              {/* Detailed Stats Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* General Stats */}
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700 pb-2">General Statistics</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                      <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Missing</span>
                      <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.missing_count} ({selectedVariable.missing_percentage.toFixed(2)}%)</span>
                    </div>
                  </div>
                </div>

                {/* Type Specific Stats */}
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700 pb-2">
                    {selectedVariable.dtype} Statistics
                  </h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {selectedVariable.numeric_stats && (
                      <>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Mean</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.mean?.toFixed(4)}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Median</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.median?.toFixed(4)}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Std Dev</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.std?.toFixed(4)}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.min}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.max}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Zeros</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.zeros_count}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Skewness</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.skewness?.toFixed(4)}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Kurtosis</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.kurtosis?.toFixed(4)}</span></div>
                        {selectedVariable.normality_test && (
                          <>
                            <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                                <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Normality ({selectedVariable.normality_test.test_name})</span>
                                <span className={`font-medium ${selectedVariable.normality_test.is_normal ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'}`}>
                                    {selectedVariable.normality_test.is_normal ? 'Normal' : 'Not Normal'}
                                </span>
                            </div>
                            <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                                <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">P-Value</span>
                                <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.normality_test.p_value.toFixed(4)}</span>
                            </div>
                          </>
                        )}
                      </>
                    )}
                    {selectedVariable.categorical_stats && (
                      <>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Unique Values</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.categorical_stats.unique_count}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Rare Labels</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.categorical_stats.rare_labels_count}</span></div>
                      </>
                    )}
                    {selectedVariable.text_stats && (
                      <>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Avg Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.avg_length?.toFixed(2)}</span></div>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.min_length}</span></div>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.max_length}</span></div>
                      </>
                    )}
                    {selectedVariable.date_stats && (
                      <>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min Date</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.min_date}</span></div>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max Date</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.max_date}</span></div>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Duration (Days)</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.duration_days?.toFixed(1)}</span></div>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Additional Lists (Top K / Common Words) */}
              {selectedVariable.categorical_stats?.top_k && (
                  <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800">
                      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3 uppercase tracking-wider">Top Categories</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {selectedVariable.categorical_stats.top_k.map((item: any, idx: number) => (
                              <div key={idx} className="flex justify-between items-center p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-sm">
                                  <span className="truncate font-medium text-gray-700 dark:text-gray-300" title={item.value}>{item.value}</span>
                                  <span className="text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-900 px-2 py-0.5 rounded-full text-xs">{item.count}</span>
                              </div>
                          ))}
                      </div>
                  </div>
              )}

              {selectedVariable.text_stats?.common_words && (
                  <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800">
                      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3 uppercase tracking-wider">Most Common Words</h3>
                      <div className="flex flex-wrap gap-2">
                          {selectedVariable.text_stats.common_words.map((item: any, idx: number) => (
                              <div key={idx} className="flex items-center p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-sm">
                                  <span className="font-medium text-blue-600 dark:text-blue-400 mr-2">{item.word}</span>
                                  <span className="text-gray-500 dark:text-gray-400 text-xs">({item.count})</span>
                              </div>
                          ))}
                      </div>
                  </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      <JobsHistoryModal 
        isOpen={showHistoryModal}
        onClose={() => setShowHistoryModal(false)}
        history={history}
        onFetchReport={async (id) => {
            return await EDAService.getReport(id);
        }}
        onSelect={(selectedReport) => {
            setReport(selectedReport);
            // Also update excluded cols state to match the report
            if (selectedReport.profile_data?.excluded_columns) {
                setExcludedCols(selectedReport.profile_data.excluded_columns);
            } else {
                setExcludedCols([]);
            }
        }}
      />
    </div>
  );
};
