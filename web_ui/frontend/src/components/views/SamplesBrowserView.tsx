import { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Folder, FolderOpen, Image, RefreshCw } from 'lucide-react';
import { samplesApi, type TreeNode, type SampleImage } from '../../lib/api';

interface SamplesBrowserViewProps {
    samplesDir?: string;
}

export function SamplesBrowserView({ samplesDir }: SamplesBrowserViewProps) {
    const [tree, setTree] = useState<TreeNode[]>([]);
    const [selectedPath, setSelectedPath] = useState<string | null>(null);
    const [selectedName, setSelectedName] = useState<string>('');
    const [images, setImages] = useState<SampleImage[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingImages, setLoadingImages] = useState(false);
    const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
    const [selectedImage, setSelectedImage] = useState<SampleImage | null>(null);

    // Fetch tree on mount
    useEffect(() => {
        fetchTree();
    }, [samplesDir]);

    const fetchTree = async () => {
        setLoading(true);
        try {
            const response = await samplesApi.getTree(samplesDir);
            setTree(response.data.tree);
        } catch (err) {
            console.error('Failed to fetch samples tree:', err);
        } finally {
            setLoading(false);
        }
    };

    const fetchImages = async (path: string) => {
        setLoadingImages(true);
        try {
            const response = await samplesApi.listImages(path);
            setImages(response.data.images);
        } catch (err) {
            console.error('Failed to fetch images:', err);
            setImages([]);
        } finally {
            setLoadingImages(false);
        }
    };

    const toggleNode = (path: string) => {
        const newExpanded = new Set(expandedNodes);
        if (newExpanded.has(path)) {
            newExpanded.delete(path);
        } else {
            newExpanded.add(path);
        }
        setExpandedNodes(newExpanded);
    };

    const selectNode = (node: TreeNode) => {
        setSelectedPath(node.path);
        setSelectedName(node.name);
        if (node.type === 'prompt' || node.type === 'directory') {
            fetchImages(node.path);
        }
    };

    const renderTreeNode = (node: TreeNode, depth: number = 0) => {
        const isExpanded = expandedNodes.has(node.path);
        const isSelected = selectedPath === node.path;
        const hasChildren = node.children && node.children.length > 0;

        return (
            <div key={node.path}>
                <div
                    className={`flex items-center gap-1 py-1 px-2 cursor-pointer hover:bg-dark-hover transition-colors text-sm
            ${isSelected ? 'bg-primary/20 text-primary' : 'text-white'}`}
                    style={{ paddingLeft: `${depth * 16 + 8}px` }}
                    onClick={() => {
                        if (hasChildren) toggleNode(node.path);
                        selectNode(node);
                    }}
                >
                    {/* Expand/Collapse Icon */}
                    {hasChildren ? (
                        isExpanded ? (
                            <ChevronDown className="w-3 h-3 text-muted flex-shrink-0" />
                        ) : (
                            <ChevronRight className="w-3 h-3 text-muted flex-shrink-0" />
                        )
                    ) : (
                        <span className="w-3" />
                    )}

                    {/* Folder/Prompt Icon */}
                    {node.type === 'directory' ? (
                        isExpanded ? (
                            <FolderOpen className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                        ) : (
                            <Folder className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                        )
                    ) : (
                        <Image className="w-4 h-4 text-blue-400 flex-shrink-0" />
                    )}

                    {/* Name */}
                    <span className="truncate flex-1">{node.name}</span>

                    {/* Image Count Badge */}
                    {node.image_count && node.image_count > 0 && (
                        <span className="px-1.5 py-0.5 bg-dark-border rounded text-xs text-muted">
                            {node.image_count}
                        </span>
                    )}
                </div>

                {/* Children */}
                {hasChildren && isExpanded && (
                    <div>
                        {node.children!.map((child) => renderTreeNode(child, depth + 1))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="h-full flex">
            {/* Left Panel - Tree */}
            <div className="w-64 border-r border-dark-border bg-dark-surface flex flex-col">
                <div className="p-3 border-b border-dark-border flex items-center justify-between">
                    <span className="text-sm font-medium text-muted uppercase tracking-wider">Samples</span>
                    <button
                        onClick={fetchTree}
                        className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white"
                        title="Refresh"
                    >
                        <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto">
                    {loading ? (
                        <div className="p-4 text-center text-muted text-sm">Loading...</div>
                    ) : tree.length === 0 ? (
                        <div className="p-4 text-center text-muted text-sm">
                            No sample directories found
                        </div>
                    ) : (
                        tree.map((node) => renderTreeNode(node))
                    )}
                </div>
            </div>

            {/* Right Panel - Image Gallery */}
            <div className="flex-1 flex flex-col bg-dark-bg">
                {/* Header */}
                <div className="h-12 px-4 flex items-center border-b border-dark-border bg-dark-surface">
                    <span className="text-sm text-white font-medium">
                        {selectedName || 'Select a prompt to view samples'}
                    </span>
                    {images.length > 0 && (
                        <span className="ml-2 text-xs text-muted">({images.length} images)</span>
                    )}
                </div>

                {/* Gallery */}
                <div className="flex-1 overflow-y-auto p-4">
                    {!selectedPath ? (
                        <div className="flex flex-col items-center justify-center h-full text-muted">
                            <Image className="w-16 h-16 mb-4 opacity-20" />
                            <p className="text-sm">Select a prompt from the tree to view samples</p>
                        </div>
                    ) : loadingImages ? (
                        <div className="flex items-center justify-center h-full text-muted">
                            <RefreshCw className="w-6 h-6 animate-spin" />
                        </div>
                    ) : images.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-muted">
                            <Image className="w-16 h-16 mb-4 opacity-20" />
                            <p className="text-sm">No images in this folder</p>
                        </div>
                    ) : (
                        <div className="flex flex-wrap gap-2">
                            {images.map((image) => (
                                <div
                                    key={image.id}
                                    className="relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all border-transparent hover:border-primary/50"
                                    onClick={() => setSelectedImage(image)}
                                >
                                    <img
                                        src={`/api/samples/${encodeURIComponent(image.id)}?samples_dir=${encodeURIComponent(selectedPath!)}`}
                                        alt={image.name}
                                        className="w-32 h-32 object-cover bg-dark-surface"
                                        loading="lazy"
                                        onError={(e) => {
                                            // Fallback to direct path
                                            (e.target as HTMLImageElement).src = `/api/filesystem/file?path=${encodeURIComponent(image.path)}`;
                                        }}
                                    />
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Lightbox Modal for Full-size Image */}
            {selectedImage && (
                <div
                    className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center cursor-pointer"
                    onClick={() => setSelectedImage(null)}
                    onKeyDown={(e) => e.key === 'Escape' && setSelectedImage(null)}
                    tabIndex={0}
                >
                    <div className="absolute top-4 right-4 text-white text-sm bg-black/50 px-3 py-1 rounded">
                        Click anywhere to close
                    </div>
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white text-sm bg-black/50 px-3 py-1 rounded">
                        {selectedImage.name}
                    </div>
                    <img
                        src={`/api/filesystem/file?path=${encodeURIComponent(selectedImage.path)}`}
                        alt={selectedImage.name}
                        className="max-h-[90vh] max-w-[90vw] object-contain"
                        onClick={(e) => e.stopPropagation()}
                    />
                </div>
            )}
        </div>
    );
}
