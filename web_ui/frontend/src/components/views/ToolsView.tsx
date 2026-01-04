import { useState } from 'react';
import { CaptionerTool } from '../tools/CaptionerTool';
import { ImageTools } from '../tools/ImageTools';

interface ConversionSettings {
  source_path: string;
  output_path: string;
  source_format: string;
  target_format: string;
}

interface MaskSettings {
  source_dir: string;
  mask_model: string;
  threshold: number;
  invert_mask: boolean;
}

interface DatasetSettings {
  source_dir: string;
  output_dir: string;
  target_resolution: number;
  resize_mode: string;
}

export function ToolsView() {
  const [conversionSettings, setConversionSettings] = useState<ConversionSettings>({
    source_path: '',
    output_path: '',
    source_format: 'safetensors',
    target_format: 'diffusers',
  });

  const [maskSettings, setMaskSettings] = useState<MaskSettings>({
    source_dir: '',
    mask_model: 'segment_anything',
    threshold: 0.5,
    invert_mask: false,
  });

  const [datasetSettings, setDatasetSettings] = useState<DatasetSettings>({
    source_dir: '',
    output_dir: '',
    target_resolution: 512,
    resize_mode: 'resize',
  });

  /* Tab State */
  const [activeTab, setActiveTab] = useState<'conversion' | 'caption' | 'mask' | 'dataset' | 'image'>('caption');

  return (
    <div className="h-full flex flex-col bg-dark-bg">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-dark-border bg-dark-surface sticky top-0 z-10">
        <h1 className="text-lg font-medium text-white">Utility Tools</h1>

        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-dark-bg p-1 rounded-lg border border-dark-border">
          <button
            onClick={() => setActiveTab('caption')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'caption'
              ? 'bg-primary text-white shadow-sm'
              : 'text-muted hover:text-white hover:bg-dark-hover'
              }`}
          >
            Captioner
          </button>
          <button
            onClick={() => setActiveTab('conversion')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'conversion'
              ? 'bg-primary text-white shadow-sm'
              : 'text-muted hover:text-white hover:bg-dark-hover'
              }`}
          >
            Model Conversion
          </button>
          <button
            onClick={() => setActiveTab('mask')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'mask'
              ? 'bg-primary text-white shadow-sm'
              : 'text-muted hover:text-white hover:bg-dark-hover'
              }`}
          >
            Mask Generation
          </button>
          <button
            onClick={() => setActiveTab('dataset')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'dataset'
              ? 'bg-primary text-white shadow-sm'
              : 'text-muted hover:text-white hover:bg-dark-hover'
              }`}
          >
            Dataset Tools
          </button>
          <button
            onClick={() => setActiveTab('image')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'image'
              ? 'bg-primary text-white shadow-sm'
              : 'text-muted hover:text-white hover:bg-dark-hover'
              }`}
          >
            Image Tools
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="h-full">
          {/* Captioner Tool */}
          {activeTab === 'caption' && <CaptionerTool />}

          {/* Image Tools */}
          {activeTab === 'image' && <ImageTools />}

          {/* Model Conversion */}
          {activeTab === 'conversion' && (
            <div className="max-w-3xl space-y-6">
              <div className="bg-dark-surface rounded-lg border border-dark-border">
                <div className="px-4 py-3 border-b border-dark-border">
                  <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Model Conversion</h2>
                </div>
                <div className="p-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-muted block mb-2">Source Format</label>
                      <select
                        value={conversionSettings.source_format}
                        onChange={(e) => setConversionSettings({ ...conversionSettings, source_format: e.target.value })}
                        className="input w-full"
                      >
                        <option value="safetensors">SafeTensors</option>
                        <option value="checkpoint">Checkpoint (.ckpt)</option>
                        <option value="diffusers">Diffusers</option>
                        <option value="kohya">Kohya</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-sm text-muted block mb-2">Target Format</label>
                      <select
                        value={conversionSettings.target_format}
                        onChange={(e) => setConversionSettings({ ...conversionSettings, target_format: e.target.value })}
                        className="input w-full"
                      >
                        <option value="safetensors">SafeTensors</option>
                        <option value="checkpoint">Checkpoint (.ckpt)</option>
                        <option value="diffusers">Diffusers</option>
                        <option value="kohya">Kohya</option>
                      </select>
                    </div>
                  </div>
                  <div>
                    <label className="text-sm text-muted block mb-2">Source Path</label>
                    <input
                      type="text"
                      value={conversionSettings.source_path}
                      onChange={(e) => setConversionSettings({ ...conversionSettings, source_path: e.target.value })}
                      className="input w-full"
                      placeholder="/path/to/source/model.safetensors"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-muted block mb-2">Output Path</label>
                    <input
                      type="text"
                      value={conversionSettings.output_path}
                      onChange={(e) => setConversionSettings({ ...conversionSettings, output_path: e.target.value })}
                      className="input w-full"
                      placeholder="/path/to/output"
                    />
                  </div>
                  <div className="flex gap-3 pt-2">
                    <button className="px-4 py-2 bg-primary hover:bg-primary-light text-white rounded-md transition-colors font-medium">
                      Convert Model
                    </button>
                    <button className="px-4 py-2 bg-dark-border hover:bg-dark-hover text-white rounded-md transition-colors">
                      Merge Models
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Mask Generation */}
          {activeTab === 'mask' && (
            <div className="max-w-3xl space-y-6">
              <div className="bg-dark-surface rounded-lg border border-dark-border">
                <div className="px-4 py-3 border-b border-dark-border">
                  <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Mask Generation</h2>
                </div>
                <div className="p-4 space-y-4">
                  <div>
                    <label className="text-sm text-muted block mb-2">Image Directory</label>
                    <input
                      type="text"
                      value={maskSettings.source_dir}
                      onChange={(e) => setMaskSettings({ ...maskSettings, source_dir: e.target.value })}
                      className="input w-full"
                      placeholder="/path/to/images"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-muted block mb-2">Mask Model</label>
                      <select
                        value={maskSettings.mask_model}
                        onChange={(e) => setMaskSettings({ ...maskSettings, mask_model: e.target.value })}
                        className="input w-full"
                      >
                        <option value="segment_anything">Segment Anything (SAM)</option>
                        <option value="rembg">RemBG</option>
                        <option value="clipseg">CLIPSeg</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-sm text-muted block mb-2">Threshold</label>
                      <input
                        type="text"
                        value={maskSettings.threshold}
                        onChange={(e) => setMaskSettings({ ...maskSettings, threshold: parseFloat(e.target.value) || 0.5 })}
                        className="input w-full"
                        min="0"
                        max="1"
                        step="0.1"
                      />
                    </div>
                  </div>
                  <div className="flex items-center">
                    <label className="flex items-center gap-2 text-sm text-white cursor-pointer">
                      <input
                        type="checkbox"
                        checked={maskSettings.invert_mask}
                        onChange={(e) => setMaskSettings({ ...maskSettings, invert_mask: e.target.checked })}
                        className="rounded border-dark-border"
                      />
                      Invert Mask
                    </label>
                  </div>
                  <div className="flex gap-3 pt-2">
                    <button className="px-4 py-2 bg-primary hover:bg-primary-light text-white rounded-md transition-colors font-medium">
                      Generate Masks
                    </button>
                    <button className="px-4 py-2 bg-dark-border hover:bg-dark-hover text-white rounded-md transition-colors">
                      Edit Masks
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Dataset Processing */}
          {activeTab === 'dataset' && (
            <div className="max-w-3xl space-y-6">
              <div className="bg-dark-surface rounded-lg border border-dark-border">
                <div className="px-4 py-3 border-b border-dark-border">
                  <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Dataset Processing</h2>
                </div>
                <div className="p-4 space-y-4">
                  <div>
                    <label className="text-sm text-muted block mb-2">Source Directory</label>
                    <input
                      type="text"
                      value={datasetSettings.source_dir}
                      onChange={(e) => setDatasetSettings({ ...datasetSettings, source_dir: e.target.value })}
                      className="input w-full"
                      placeholder="/path/to/source/images"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-muted block mb-2">Output Directory</label>
                    <input
                      type="text"
                      value={datasetSettings.output_dir}
                      onChange={(e) => setDatasetSettings({ ...datasetSettings, output_dir: e.target.value })}
                      className="input w-full"
                      placeholder="/path/to/output"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-muted block mb-2">Target Resolution</label>
                      <select
                        value={datasetSettings.target_resolution}
                        onChange={(e) => setDatasetSettings({ ...datasetSettings, target_resolution: parseInt(e.target.value) })}
                        className="input w-full"
                      >
                        <option value={256}>256px</option>
                        <option value={512}>512px</option>
                        <option value={768}>768px</option>
                        <option value={1024}>1024px</option>
                        <option value={1280}>1280px</option>
                        <option value={1536}>1536px</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-sm text-muted block mb-2">Resize Mode</label>
                      <select
                        value={datasetSettings.resize_mode}
                        onChange={(e) => setDatasetSettings({ ...datasetSettings, resize_mode: e.target.value })}
                        className="input w-full"
                      >
                        <option value="resize">Resize (stretch)</option>
                        <option value="crop">Center Crop</option>
                        <option value="pad">Pad (letterbox)</option>
                        <option value="fill">Fill (crop to fit)</option>
                      </select>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3 pt-2">
                    <button className="px-4 py-2 bg-primary hover:bg-primary-light text-white rounded-md transition-colors font-medium">
                      Process Images
                    </button>
                    <button className="px-4 py-2 bg-dark-border hover:bg-dark-hover text-white rounded-md transition-colors">
                      Analyze Dataset
                    </button>
                    <button className="px-4 py-2 bg-dark-border hover:bg-dark-hover text-white rounded-md transition-colors">
                      Remove Duplicates
                    </button>
                    <button className="px-4 py-2 bg-dark-border hover:bg-dark-hover text-white rounded-md transition-colors">
                      Clean Dataset
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
