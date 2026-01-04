import { useState, useEffect, useCallback } from 'react';
import {
  FolderOpen,
  Image as ImageIcon,
  FileText,
  ChevronLeft,
  ChevronRight,
  X,
  RefreshCw,
  Grid,
  List,
  Search,
  Folder,
  Info,
  Database,
  ExternalLink
} from 'lucide-react';
import { filesystemApi } from '../../lib/api';
import { useConfigStore } from '../../stores/configStore';

const DEFAULT_DATASETS_DIR = '/home/alex/OneTrainer/training_concepts/';

interface DatasetEntry {
  name: string;
  path: string;
  is_directory: boolean;
  size: number | null;
}

interface ImageFile {
  path: string;
  filename: string;
  size: number | null;
  width: number | null;
  height: number | null;
  caption?: string;
}

export function DatasetsView() {
  // Config store
  const { config, currentPreset } = useConfigStore();
  const concepts = (config as any)?.concepts || [];

  // Navigation state
  const [currentPath, setCurrentPath] = useState(DEFAULT_DATASETS_DIR);
  const [breadcrumbs, setBreadcrumbs] = useState<{ name: string; path: string }[]>([]);
  const [showCurrentDataset, setShowCurrentDataset] = useState(true);

  // Dataset listing state
  const [directories, setDirectories] = useState<DatasetEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Image browsing state
  const [images, setImages] = useState<ImageFile[]>([]);
  const [loadingImages, setLoadingImages] = useState(false);
  const [selectedImage, setSelectedImage] = useState<ImageFile | null>(null);
  const [imageIndex, setImageIndex] = useState(0);

  // View options
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchFilter, setSearchFilter] = useState('');
  const [showInfo, setShowInfo] = useState(true);

  // Load directories on path change
  useEffect(() => {
    loadDirectory(currentPath);
    updateBreadcrumbs(currentPath);
  }, [currentPath]);

  const updateBreadcrumbs = (path: string) => {
    const parts = path.replace(DEFAULT_DATASETS_DIR, '').split('/').filter(Boolean);
    const crumbs = [{ name: 'Datasets', path: DEFAULT_DATASETS_DIR }];
    let accPath = DEFAULT_DATASETS_DIR;

    for (const part of parts) {
      accPath = accPath.endsWith('/') ? accPath + part : accPath + '/' + part;
      crumbs.push({ name: part, path: accPath });
    }

    setBreadcrumbs(crumbs);
  };

  const loadDirectory = async (path: string) => {
    try {
      setLoading(true);
      setError(null);

      // Browse for directories and files
      const response = await filesystemApi.browse(path);
      const entries = response.data.entries as DatasetEntry[];

      // Separate directories
      const dirs = entries.filter(e => e.is_directory);
      setDirectories(dirs);

      // Scan for images
      await scanForImages(path);

    } catch (err: any) {
      console.error('Failed to load directory:', err);
      setError(err.response?.data?.detail || 'Failed to load directory');
    } finally {
      setLoading(false);
    }
  };

  const scanForImages = async (path: string) => {
    try {
      setLoadingImages(true);

      // Scan for images with dimensions
      const response = await filesystemApi.scan(path, false, true, 500);
      const imageFiles: ImageFile[] = response.data.files || [];

      // Load captions for each image
      const imagesWithCaptions = await Promise.all(
        imageFiles.map(async (img) => {
          const caption = await loadCaption(img.path);
          return { ...img, caption };
        })
      );

      setImages(imagesWithCaptions);
    } catch (err) {
      console.error('Failed to scan images:', err);
      setImages([]);
    } finally {
      setLoadingImages(false);
    }
  };

  const loadCaption = async (imagePath: string): Promise<string | undefined> => {
    // Try different caption file extensions
    const basePath = imagePath.replace(/\.[^.]+$/, '');
    const captionExtensions = ['.txt', '.caption', '.cap'];

    for (const ext of captionExtensions) {
      try {
        const response = await fetch(`/api/filesystem/file?path=${encodeURIComponent(basePath + ext)}`);
        if (response.ok) {
          return await response.text();
        }
      } catch {
        // Caption file doesn't exist, continue
      }
    }
    return undefined;
  };

  const navigateToDirectory = (path: string) => {
    setCurrentPath(path);
    setSelectedImage(null);
  };

  const goUp = () => {
    const parentPath = currentPath.replace(/\/[^/]+\/?$/, '');
    if (parentPath.startsWith(DEFAULT_DATASETS_DIR.replace(/\/$/, ''))) {
      setCurrentPath(parentPath || DEFAULT_DATASETS_DIR);
    }
  };

  const handleImageClick = (image: ImageFile, index: number) => {
    setSelectedImage(image);
    setImageIndex(index);
  };

  const navigateImage = useCallback((direction: 'prev' | 'next') => {
    const filteredImages = getFilteredImages();
    if (filteredImages.length === 0) return;

    let newIndex = imageIndex;
    if (direction === 'prev') {
      newIndex = imageIndex > 0 ? imageIndex - 1 : filteredImages.length - 1;
    } else {
      newIndex = imageIndex < filteredImages.length - 1 ? imageIndex + 1 : 0;
    }

    setImageIndex(newIndex);
    setSelectedImage(filteredImages[newIndex]);
  }, [imageIndex, images, searchFilter]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!selectedImage) return;

      if (e.key === 'ArrowLeft') {
        navigateImage('prev');
      } else if (e.key === 'ArrowRight') {
        navigateImage('next');
      } else if (e.key === 'Escape') {
        setSelectedImage(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedImage, navigateImage]);

  const getFilteredImages = () => {
    if (!searchFilter) return images;
    const lower = searchFilter.toLowerCase();
    return images.filter(img =>
      img.filename.toLowerCase().includes(lower) ||
      img.caption?.toLowerCase().includes(lower)
    );
  };

  const filteredImages = getFilteredImages();
  const imageUrl = (path: string) => `/api/filesystem/file?path=${encodeURIComponent(path)}`;

  const formatSize = (bytes: number | null) => {
    if (!bytes) return '--';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-dark-border bg-dark-surface">
        <div className="flex items-center gap-3">
          <FolderOpen className="w-5 h-5 text-primary" />
          <h1 className="text-lg font-medium text-white">Datasets</h1>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => loadDirectory(currentPath)}
            className="p-2 hover:bg-dark-hover rounded text-muted hover:text-white"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 ${loading || loadingImages ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Breadcrumb Navigation */}
      <div className="px-6 py-2 border-b border-dark-border bg-dark-surface/50 flex items-center gap-2 text-sm">
        {breadcrumbs.map((crumb, idx) => (
          <div key={crumb.path} className="flex items-center gap-2">
            {idx > 0 && <span className="text-muted">/</span>}
            <button
              onClick={() => navigateToDirectory(crumb.path)}
              className={`hover:text-primary ${idx === breadcrumbs.length - 1 ? 'text-white font-medium' : 'text-muted'}`}
            >
              {crumb.name}
            </button>
          </div>
        ))}
      </div>

      {/* Toolbar */}
      <div className="px-6 py-3 border-b border-dark-border bg-dark-surface/30 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
            <input
              type="text"
              placeholder="Search images or captions..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              className="pl-9 pr-4 py-1.5 bg-dark-bg border border-dark-border rounded text-sm text-white placeholder:text-muted w-64"
            />
          </div>

          {/* Stats */}
          <div className="text-sm text-muted">
            {directories.length > 0 && (
              <span className="mr-4">{directories.length} folder{directories.length !== 1 ? 's' : ''}</span>
            )}
            <span>{filteredImages.length} image{filteredImages.length !== 1 ? 's' : ''}</span>
            {searchFilter && images.length !== filteredImages.length && (
              <span className="text-primary ml-1">(filtered from {images.length})</span>
            )}
          </div>
        </div>

        {/* View toggle */}
        <div className="flex items-center gap-1 bg-dark-bg rounded p-0.5">
          <button
            onClick={() => setViewMode('grid')}
            className={`p-1.5 rounded ${viewMode === 'grid' ? 'bg-dark-hover text-white' : 'text-muted hover:text-white'}`}
            title="Grid view"
          >
            <Grid className="w-4 h-4" />
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`p-1.5 rounded ${viewMode === 'list' ? 'bg-dark-hover text-white' : 'text-muted hover:text-white'}`}
            title="List view"
          >
            <List className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Current Dataset from Preset */}
      {showCurrentDataset && concepts.length > 0 && (
        <div className="mx-6 mt-4 p-4 bg-dark-surface border border-dark-border rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-primary" />
              <h3 className="text-sm font-medium text-white">Current Dataset</h3>
              {currentPreset && (
                <span className="text-xs text-muted px-2 py-0.5 bg-dark-bg rounded">from: {currentPreset}</span>
              )}
            </div>
            <button onClick={() => setShowCurrentDataset(false)} className="text-muted hover:text-white">
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="space-y-2">
            {concepts.map((concept: any, idx: number) => (
              <div key={idx} className={`flex items-center justify-between p-2 rounded ${concept.enabled !== false ? 'bg-dark-bg' : 'bg-dark-bg/50 opacity-60'}`}>
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`w-2 h-2 rounded-full ${concept.enabled !== false ? 'bg-green-500' : 'bg-gray-500'}`} />
                  <div className="min-w-0">
                    <p className="text-sm text-white truncate">{concept.name || concept.path?.split('/').pop() || `Concept ${idx + 1}`}</p>
                    <p className="text-xs text-muted truncate">{concept.path || 'No path set'}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted px-2 py-0.5 bg-dark-border rounded">{concept.type || 'STANDARD'}</span>
                  {concept.path && (
                    <button
                      onClick={() => navigateToDirectory(concept.path)}
                      className="p-1 text-muted hover:text-primary"
                      title="Browse this path"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
          {concepts.length === 0 && (
            <p className="text-sm text-muted text-center py-2">No concepts configured. Load a preset or add concepts.</p>
          )}
        </div>
      )}

      {/* Show current dataset button when hidden */}
      {!showCurrentDataset && concepts.length > 0 && (
        <div className="mx-6 mt-4">
          <button
            onClick={() => setShowCurrentDataset(true)}
            className="text-sm text-primary hover:text-primary-hover flex items-center gap-2"
          >
            <Database className="w-4 h-4" />
            Show current dataset ({concepts.length} concept{concepts.length !== 1 ? 's' : ''})
          </button>
        </div>
      )}

      {/* No preset loaded message */}
      {concepts.length === 0 && (
        <div className="mx-6 mt-4 p-4 bg-dark-surface border border-dark-border rounded-lg">
          <div className="flex items-center gap-2 text-muted">
            <Database className="w-4 h-4" />
            <span className="text-sm">No preset loaded. Load a preset from "New Job" to see dataset concepts.</span>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-6">
        {/* Error State */}
        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 mb-4">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12 text-muted">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-3" />
            <p>Loading...</p>
          </div>
        )}

        {!loading && !error && (
          <>
            {/* Directories */}
            {directories.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-muted mb-3">Folders</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                  {currentPath !== DEFAULT_DATASETS_DIR && (
                    <button
                      onClick={goUp}
                      className="flex items-center gap-2 p-3 bg-dark-surface border border-dark-border rounded-lg hover:border-primary transition-colors"
                    >
                      <ChevronLeft className="w-5 h-5 text-muted" />
                      <span className="text-sm text-muted">Go up</span>
                    </button>
                  )}
                  {directories.map((dir) => (
                    <button
                      key={dir.path}
                      onClick={() => navigateToDirectory(dir.path)}
                      className="flex items-center gap-2 p-3 bg-dark-surface border border-dark-border rounded-lg hover:border-primary transition-colors text-left"
                    >
                      <Folder className="w-5 h-5 text-yellow-500 flex-shrink-0" />
                      <span className="text-sm text-white truncate">{dir.name}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Images */}
            {filteredImages.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-muted mb-3">Images</h3>

                {viewMode === 'grid' ? (
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                    {filteredImages.map((img, idx) => (
                      <div
                        key={img.path}
                        onClick={() => handleImageClick(img, idx)}
                        className="group cursor-pointer bg-dark-surface border border-dark-border rounded-lg overflow-hidden hover:border-primary transition-colors"
                      >
                        <div className="aspect-square relative overflow-hidden bg-dark-bg">
                          <img
                            src={imageUrl(img.path)}
                            alt={img.filename}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                            loading="lazy"
                          />
                          {img.caption && (
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                              <FileText className="w-3 h-3 text-green-400" />
                            </div>
                          )}
                        </div>
                        <div className="p-2">
                          <p className="text-xs text-white truncate" title={img.filename}>
                            {img.filename}
                          </p>
                          {img.width && img.height && (
                            <p className="text-xs text-muted">
                              {img.width} × {img.height}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-2">
                    {filteredImages.map((img, idx) => (
                      <div
                        key={img.path}
                        onClick={() => handleImageClick(img, idx)}
                        className="flex items-center gap-4 p-3 bg-dark-surface border border-dark-border rounded-lg hover:border-primary cursor-pointer transition-colors"
                      >
                        <div className="w-16 h-16 flex-shrink-0 bg-dark-bg rounded overflow-hidden">
                          <img
                            src={imageUrl(img.path)}
                            alt={img.filename}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-white truncate">{img.filename}</p>
                          <div className="flex items-center gap-4 text-xs text-muted mt-1">
                            {img.width && img.height && (
                              <span>{img.width} × {img.height}</span>
                            )}
                            <span>{formatSize(img.size)}</span>
                            {img.caption && (
                              <span className="text-green-400 flex items-center gap-1">
                                <FileText className="w-3 h-3" />
                                Has caption
                              </span>
                            )}
                          </div>
                          {img.caption && (
                            <p className="text-xs text-muted mt-1 truncate">
                              {img.caption}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Empty State */}
            {directories.length === 0 && filteredImages.length === 0 && !loadingImages && (
              <div className="text-center py-12">
                <ImageIcon className="w-16 h-16 mx-auto mb-4 text-muted opacity-30" />
                <p className="text-muted">No images found in this folder</p>
                <p className="text-sm text-muted mt-1">
                  Add images to your training concepts folder
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Image Viewer Modal */}
      {selectedImage && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center">
          {/* Close button */}
          <button
            onClick={() => setSelectedImage(null)}
            className="absolute top-4 right-4 p-2 text-white/70 hover:text-white bg-black/50 rounded-lg"
          >
            <X className="w-6 h-6" />
          </button>

          {/* Navigation */}
          <button
            onClick={() => navigateImage('prev')}
            className="absolute left-4 top-1/2 -translate-y-1/2 p-3 text-white/70 hover:text-white bg-black/50 rounded-lg"
          >
            <ChevronLeft className="w-8 h-8" />
          </button>
          <button
            onClick={() => navigateImage('next')}
            className="absolute right-4 top-1/2 -translate-y-1/2 p-3 text-white/70 hover:text-white bg-black/50 rounded-lg"
          >
            <ChevronRight className="w-8 h-8" />
          </button>

          {/* Main content */}
          <div className="flex max-w-[90vw] max-h-[90vh] gap-4">
            {/* Image */}
            <div className="flex-1 flex items-center justify-center">
              <img
                src={imageUrl(selectedImage.path)}
                alt={selectedImage.filename}
                className="max-w-full max-h-[85vh] object-contain rounded-lg"
              />
            </div>

            {/* Info Panel */}
            {showInfo && (
              <div className="w-80 bg-dark-surface rounded-lg p-4 flex flex-col max-h-[85vh]">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-white">Image Details</h3>
                  <button
                    onClick={() => setShowInfo(false)}
                    className="p-1 text-muted hover:text-white"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                <div className="space-y-3 text-sm">
                  <div>
                    <label className="text-muted text-xs">Filename</label>
                    <p className="text-white break-all">{selectedImage.filename}</p>
                  </div>

                  {selectedImage.width && selectedImage.height && (
                    <div>
                      <label className="text-muted text-xs">Dimensions</label>
                      <p className="text-white">{selectedImage.width} × {selectedImage.height}</p>
                    </div>
                  )}

                  <div>
                    <label className="text-muted text-xs">Size</label>
                    <p className="text-white">{formatSize(selectedImage.size)}</p>
                  </div>

                  <div>
                    <label className="text-muted text-xs">Index</label>
                    <p className="text-white">{imageIndex + 1} of {filteredImages.length}</p>
                  </div>
                </div>

                {/* Caption */}
                <div className="mt-4 flex-1 min-h-0">
                  <label className="text-muted text-xs flex items-center gap-1">
                    <FileText className="w-3 h-3" />
                    Caption
                  </label>
                  <div className="mt-1 p-3 bg-dark-bg rounded-lg overflow-auto max-h-64">
                    {selectedImage.caption ? (
                      <p className="text-white text-sm whitespace-pre-wrap">{selectedImage.caption}</p>
                    ) : (
                      <p className="text-muted text-sm italic">No caption file found</p>
                    )}
                  </div>
                </div>

                {/* Path */}
                <div className="mt-4 pt-4 border-t border-dark-border">
                  <label className="text-muted text-xs">Full Path</label>
                  <p className="text-white/60 text-xs break-all mt-1">{selectedImage.path}</p>
                </div>
              </div>
            )}
          </div>

          {/* Show info button (when hidden) */}
          {!showInfo && (
            <button
              onClick={() => setShowInfo(true)}
              className="absolute bottom-4 right-4 p-2 text-white/70 hover:text-white bg-black/50 rounded-lg flex items-center gap-2"
            >
              <Info className="w-5 h-5" />
              <span className="text-sm">Show Info</span>
            </button>
          )}

          {/* Image counter */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 bg-black/50 rounded-lg text-white text-sm">
            {imageIndex + 1} / {filteredImages.length}
          </div>
        </div>
      )}
    </div>
  );
}
