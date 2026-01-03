import Plotly from 'plotly.js-dist-min';
import html2canvas from 'html2canvas';

export const downloadChart = async (elementId: string, filename: string, title?: string, subtitle?: string, extraInfo?: string) => {
    const element = document.getElementById(elementId);
    if (!element) return;

    const isDarkMode = document.documentElement.classList.contains('dark');
    const bgColor = isDarkMode ? '#1f2937' : '#ffffff'; // gray-800 vs white
    const textColor = isDarkMode ? '#f3f4f6' : '#111827'; // gray-100 vs gray-900
    const subTextColor = isDarkMode ? '#9ca3af' : '#4b5563'; // gray-400 vs gray-600

    // Helper to draw text on canvas
    const drawText = (ctx: CanvasRenderingContext2D, width: number) => {
        let offset = 0;
        if (title) {
            ctx.fillStyle = textColor;
            ctx.font = 'bold 16px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(title, width / 2, 30);
            offset = 40;
        }
        
        if (subtitle) {
            ctx.fillStyle = subTextColor;
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(subtitle, width / 2, 50);
            offset = 60;
        }

        if (extraInfo) {
            ctx.fillStyle = textColor;
            ctx.font = '10px monospace';
            ctx.textAlign = 'right';
            const lines = extraInfo.split('\n');
            lines.forEach((line, i) => {
                ctx.fillText(line, width - 10, 20 + (i * 12));
            });
        }

        return offset || 20;
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

                    // Draw Extra Info (Top Right)
                    if (extraInfo) {
                        ctx.fillStyle = textColor;
                        ctx.font = '14px monospace';
                        ctx.textAlign = 'right';
                        const lines = extraInfo.split('\n');
                        lines.forEach((line, i) => {
                            ctx.fillText(line, width - 20, 30 + (i * 18));
                        });
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

    // Use html2canvas for everything else (Recharts, HTML tables, etc.)
    // This is much more robust than manual SVG serialization
    try {
        const canvas = await html2canvas(element, {
            backgroundColor: bgColor,
            scale: 2, // High resolution
            logging: false,
            useCORS: true
        } as any);

        const finalCanvas = document.createElement('canvas');
        const ctx = finalCanvas.getContext('2d');
        
        const headerHeight = title ? (subtitle ? 80 : 50) : 0;
        finalCanvas.width = canvas.width;
        finalCanvas.height = canvas.height + (headerHeight * 2); // Scale header height by 2 for retina

        if (ctx) {
            // Background
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);
            
            // Draw Title & Extra Info (Scaled)
            // We need to scale the drawing context to match the high-res canvas
            ctx.save();
            ctx.scale(2, 2);
            drawText(ctx, canvas.width / 2);
            ctx.restore();
            
            // Draw Chart
            ctx.drawImage(canvas, 0, headerHeight * 2);
            
            const a = document.createElement('a');
            a.download = `${filename}.png`;
            a.href = finalCanvas.toDataURL('image/png');
            a.click();
        }
    } catch (e) {
        console.error("Chart download failed", e);
    }
};
