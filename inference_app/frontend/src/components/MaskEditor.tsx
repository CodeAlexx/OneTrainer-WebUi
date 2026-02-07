/**
 * MaskEditor - Full-featured inpainting mask editor
 *
 * Features:
 * - Brush tool with adjustable size
 * - Eraser tool
 * - Lasso/polygon selection
 * - SAM2 click-to-segment
 * - Clear/invert mask
 */

import React, { useRef, useState, useEffect, useCallback } from 'react';

interface Point {
  x: number;
  y: number;
}

interface MaskEditorProps {
  image: string; // base64 image
  mask: string | null; // base64 mask
  onMaskChange: (mask: string | null) => void;
  onClose: () => void;
  width?: number;
  height?: number;
}

type Tool = 'brush' | 'eraser' | 'lasso' | 'sam2' | 'sam2_exclude';

export const MaskEditor: React.FC<MaskEditorProps> = ({
  image,
  mask,
  onMaskChange,
  onClose,
  width = 800,
  height = 600,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [tool, setTool] = useState<Tool>('brush');
  const [brushSize, setBrushSize] = useState(30);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lassoPoints, setLassoPoints] = useState<Point[]>([]);
  const [sam2Points, setSam2Points] = useState<{ point: Point; label: number }[]>([]);
  const [isLoadingSam2, setIsLoadingSam2] = useState(false);
  const [, setImageSize] = useState({ width: 0, height: 0 });
  const [scale, setScale] = useState(1);

  // Initialize canvases when image loads
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setImageSize({ width: img.width, height: img.height });

      // Calculate scale to fit in container
      const containerWidth = width;
      const containerHeight = height;
      const scaleX = containerWidth / img.width;
      const scaleY = containerHeight / img.height;
      const newScale = Math.min(scaleX, scaleY, 1);
      setScale(newScale);

      const displayWidth = Math.floor(img.width * newScale);
      const displayHeight = Math.floor(img.height * newScale);

      // Setup main canvas
      const canvas = canvasRef.current;
      const maskCanvas = maskCanvasRef.current;
      if (!canvas || !maskCanvas) return;

      canvas.width = displayWidth;
      canvas.height = displayHeight;
      maskCanvas.width = img.width;
      maskCanvas.height = img.height;

      // Draw image
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
      }

      // Initialize mask canvas (transparent = no mask, red = masked)
      const maskCtx = maskCanvas.getContext('2d');
      if (maskCtx) {
        maskCtx.clearRect(0, 0, img.width, img.height);
        // Load existing mask if provided
        if (mask) {
          const maskImg = new Image();
          maskImg.onload = () => {
            maskCtx.drawImage(maskImg, 0, 0);
            redrawCanvas();
          };
          maskImg.src = mask;
        }
      }
    };
    img.src = image;
  }, [image, mask, width, height]);

  // Redraw canvas with image + mask overlay
  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (!canvas || !maskCanvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw image
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Draw mask overlay (semi-transparent red)
      ctx.globalAlpha = 0.5;
      ctx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;

      // Draw lasso points if active
      if (lassoPoints.length > 0) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(lassoPoints[0].x, lassoPoints[0].y);
        for (let i = 1; i < lassoPoints.length; i++) {
          ctx.lineTo(lassoPoints[i].x, lassoPoints[i].y);
        }
        ctx.stroke();
      }

      // Draw SAM2 points
      sam2Points.forEach(({ point, label }) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = label === 1 ? '#00ff00' : '#ff0000';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    };
    img.src = image;
  }, [image, lassoPoints, sam2Points]);

  // Get mouse position relative to canvas
  const getMousePos = (e: React.MouseEvent): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  // Convert display coords to mask coords
  const toMaskCoords = (pos: Point): Point => ({
    x: Math.floor(pos.x / scale),
    y: Math.floor(pos.y / scale),
  });

  // Draw on mask canvas
  const drawOnMask = (pos: Point, erase: boolean = false) => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    const ctx = maskCanvas.getContext('2d');
    if (!ctx) return;

    const maskPos = toMaskCoords(pos);
    const maskBrushSize = brushSize / scale;

    ctx.beginPath();
    ctx.arc(maskPos.x, maskPos.y, maskBrushSize / 2, 0, Math.PI * 2);

    if (erase) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
    } else {
      ctx.fillStyle = 'rgba(255, 0, 0, 1)';
      ctx.fill();
    }

    redrawCanvas();
  };

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    const pos = getMousePos(e);

    if (tool === 'brush' || tool === 'eraser') {
      setIsDrawing(true);
      drawOnMask(pos, tool === 'eraser');
    } else if (tool === 'lasso') {
      setLassoPoints([pos]);
    } else if (tool === 'sam2' || tool === 'sam2_exclude') {
      const label = tool === 'sam2' ? 1 : 0;
      setSam2Points([...sam2Points, { point: pos, label }]);
      redrawCanvas();
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const pos = getMousePos(e);

    if (isDrawing && (tool === 'brush' || tool === 'eraser')) {
      drawOnMask(pos, tool === 'eraser');
    } else if (tool === 'lasso' && lassoPoints.length > 0 && e.buttons === 1) {
      setLassoPoints([...lassoPoints, pos]);
      redrawCanvas();
    }
  };

  const handleMouseUp = () => {
    if (tool === 'lasso' && lassoPoints.length > 2) {
      // Fill lasso selection
      fillLassoSelection();
    }
    setIsDrawing(false);
  };

  // Fill lasso selection on mask
  const fillLassoSelection = () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas || lassoPoints.length < 3) return;

    const ctx = maskCanvas.getContext('2d');
    if (!ctx) return;

    ctx.beginPath();
    const firstPoint = toMaskCoords(lassoPoints[0]);
    ctx.moveTo(firstPoint.x, firstPoint.y);

    for (let i = 1; i < lassoPoints.length; i++) {
      const point = toMaskCoords(lassoPoints[i]);
      ctx.lineTo(point.x, point.y);
    }

    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 0, 0, 1)';
    ctx.fill();

    setLassoPoints([]);
    redrawCanvas();
  };

  // Apply SAM2 segmentation
  const applySam2 = async () => {
    if (sam2Points.length === 0) return;

    setIsLoadingSam2(true);
    try {
      const points = sam2Points.map((p) => [
        Math.floor(p.point.x / scale),
        Math.floor(p.point.y / scale),
      ]);
      const labels = sam2Points.map((p) => p.label);

      const response = await fetch('http://localhost:8001/api/segment/point', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image, points, labels }),
      });

      const data = await response.json();
      if (data.success && data.mask) {
        // Merge SAM2 mask with current mask
        const maskCanvas = maskCanvasRef.current;
        if (maskCanvas) {
          const ctx = maskCanvas.getContext('2d');
          if (ctx) {
            const maskImg = new Image();
            maskImg.onload = () => {
              // Draw SAM2 mask in red
              ctx.globalCompositeOperation = 'source-over';

              // Create temp canvas to colorize mask
              const tempCanvas = document.createElement('canvas');
              tempCanvas.width = maskCanvas.width;
              tempCanvas.height = maskCanvas.height;
              const tempCtx = tempCanvas.getContext('2d');
              if (tempCtx) {
                tempCtx.drawImage(maskImg, 0, 0);
                const imageData = tempCtx.getImageData(
                  0,
                  0,
                  tempCanvas.width,
                  tempCanvas.height
                );
                const data = imageData.data;

                // Convert white to red
                for (let i = 0; i < data.length; i += 4) {
                  if (data[i] > 128) {
                    data[i] = 255; // R
                    data[i + 1] = 0; // G
                    data[i + 2] = 0; // B
                    data[i + 3] = 255; // A
                  } else {
                    data[i + 3] = 0; // Transparent
                  }
                }
                tempCtx.putImageData(imageData, 0, 0);
                ctx.drawImage(tempCanvas, 0, 0);
              }

              setSam2Points([]);
              redrawCanvas();
            };
            maskImg.src = data.mask;
          }
        }
      }
    } catch (error) {
      console.error('SAM2 segmentation failed:', error);
    }
    setIsLoadingSam2(false);
  };

  // Clear mask
  const clearMask = () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    const ctx = maskCanvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }
    setSam2Points([]);
    setLassoPoints([]);
    redrawCanvas();
  };

  // Invert mask
  const invertMask = () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    const ctx = maskCanvas.getContext('2d');
    if (!ctx) return;

    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      // Invert alpha based on red channel
      if (data[i] > 128) {
        // Was masked, now not
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
      } else if (data[i + 3] === 0) {
        // Was not masked, now is
        data[i] = 255;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    redrawCanvas();
  };

  // Export mask as base64
  const exportMask = (): string | null => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return null;

    const ctx = maskCanvas.getContext('2d');
    if (!ctx) return null;

    // Create grayscale mask (white = inpaint area)
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = maskCanvas.width;
    outputCanvas.height = maskCanvas.height;
    const outputCtx = outputCanvas.getContext('2d');
    if (!outputCtx) return null;

    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    // Convert red mask to white/black
    for (let i = 0; i < data.length; i += 4) {
      const isMasked = data[i] > 128 && data[i + 3] > 128;
      data[i] = isMasked ? 255 : 0;
      data[i + 1] = isMasked ? 255 : 0;
      data[i + 2] = isMasked ? 255 : 0;
      data[i + 3] = 255;
    }

    outputCtx.putImageData(imageData, 0, 0);
    return outputCanvas.toDataURL('image/png');
  };

  // Save and close
  const handleSave = () => {
    const maskData = exportMask();
    onMaskChange(maskData);
    onClose();
  };

  // Tool button component
  const ToolButton: React.FC<{
    active: boolean;
    onClick: () => void;
    title: string;
    children: React.ReactNode;
  }> = ({ active, onClick, title, children }) => (
    <button
      onClick={onClick}
      title={title}
      className={`p-2 rounded ${
        active
          ? 'bg-amber-600 text-white'
          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
      }`}
    >
      {children}
    </button>
  );

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg shadow-2xl max-w-[95vw] max-h-[95vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-white">Mask Editor</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-4 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-1.5 bg-amber-600 hover:bg-amber-500 text-white rounded"
            >
              Apply Mask
            </button>
          </div>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-4 px-4 py-2 border-b border-gray-700">
          <div className="flex items-center gap-1">
            <ToolButton
              active={tool === 'brush'}
              onClick={() => setTool('brush')}
              title="Brush (add to mask)"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M15.707 4.293a1 1 0 010 1.414l-9 9a1 1 0 01-1.414-1.414l9-9a1 1 0 011.414 0z" />
                <path d="M4 14a2 2 0 100 4 2 2 0 000-4z" />
              </svg>
            </ToolButton>
            <ToolButton
              active={tool === 'eraser'}
              onClick={() => setTool('eraser')}
              title="Eraser (remove from mask)"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M8.5 3.5a.5.5 0 00-1 0v1a.5.5 0 001 0v-1zm3 0a.5.5 0 00-1 0v1a.5.5 0 001 0v-1zm3 0a.5.5 0 00-1 0v1a.5.5 0 001 0v-1zm-9 3a.5.5 0 000 1h1a.5.5 0 000-1h-1zm12 0a.5.5 0 000 1h1a.5.5 0 000-1h-1z" />
                <path d="M6 8a2 2 0 012-2h4a2 2 0 012 2v8H6V8z" />
              </svg>
            </ToolButton>
            <ToolButton
              active={tool === 'lasso'}
              onClick={() => setTool('lasso')}
              title="Lasso selection"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M11 15l-3-3m0 0l3-3m-3 3h8M3 12a9 9 0 1118 0 9 9 0 01-18 0z"
                />
              </svg>
            </ToolButton>
          </div>

          <div className="w-px h-6 bg-gray-600" />

          <div className="flex items-center gap-1">
            <ToolButton
              active={tool === 'sam2'}
              onClick={() => setTool('sam2')}
              title="SAM2 - Click to include"
            >
              <span className="text-xs font-bold">+SAM</span>
            </ToolButton>
            <ToolButton
              active={tool === 'sam2_exclude'}
              onClick={() => setTool('sam2_exclude')}
              title="SAM2 - Click to exclude"
            >
              <span className="text-xs font-bold">-SAM</span>
            </ToolButton>
            {sam2Points.length > 0 && (
              <button
                onClick={applySam2}
                disabled={isLoadingSam2}
                className="px-3 py-1.5 bg-green-600 hover:bg-green-500 text-white rounded text-sm disabled:opacity-50"
              >
                {isLoadingSam2 ? 'Processing...' : 'Apply SAM2'}
              </button>
            )}
          </div>

          <div className="w-px h-6 bg-gray-600" />

          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Size:</span>
            <input
              type="range"
              min="5"
              max="100"
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="w-24"
            />
            <span className="text-sm text-gray-300 w-8">{brushSize}</span>
          </div>

          <div className="w-px h-6 bg-gray-600" />

          <button
            onClick={clearMask}
            className="px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white rounded text-sm"
          >
            Clear
          </button>
          <button
            onClick={invertMask}
            className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded text-sm"
          >
            Invert
          </button>
        </div>

        {/* Canvas area */}
        <div
          ref={containerRef}
          className="flex-1 overflow-auto p-4 flex items-center justify-center bg-gray-800"
        >
          <div className="relative">
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              className="cursor-crosshair border border-gray-600"
              style={{
                maxWidth: '100%',
                maxHeight: 'calc(95vh - 200px)',
              }}
            />
            {/* Hidden mask canvas */}
            <canvas ref={maskCanvasRef} className="hidden" />
          </div>
        </div>

        {/* Footer hint */}
        <div className="px-4 py-2 border-t border-gray-700 text-sm text-gray-400">
          {tool === 'brush' && 'Click and drag to paint mask area (will be regenerated)'}
          {tool === 'eraser' && 'Click and drag to erase mask area'}
          {tool === 'lasso' && 'Click and drag to draw selection, release to fill'}
          {tool === 'sam2' && 'Click on objects to include in mask (green points)'}
          {tool === 'sam2_exclude' && 'Click on objects to exclude from mask (red points)'}
        </div>
      </div>
    </div>
  );
};

export default MaskEditor;
