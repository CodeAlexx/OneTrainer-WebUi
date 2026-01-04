import { useState, useEffect } from 'react';
import { Plus, Trash2, Folder, X, Copy, Search, Image as ImageIcon } from 'lucide-react';
import { filesystemApi, configApi, databaseApi, DbConcept } from '../../lib/api';
import { useConfigStore } from '../../stores/configStore';
import { useDatabase } from '../../hooks/useDatabase';

interface ConceptImageConfig {
  enable_crop_jitter: boolean;
  enable_random_flip: boolean;
  enable_fixed_flip: boolean;
  enable_random_rotate: boolean;
  enable_fixed_rotate: boolean;
  random_rotate_max_angle: number;
  enable_random_brightness: boolean;
  enable_fixed_brightness: boolean;
  random_brightness_max_strength: number;
  enable_random_contrast: boolean;
  enable_fixed_contrast: boolean;
  random_contrast_max_strength: number;
  enable_random_saturation: boolean;
  enable_fixed_saturation: boolean;
  random_saturation_max_strength: number;
  enable_random_hue: boolean;
  enable_fixed_hue: boolean;
  random_hue_max_strength: number;
  enable_resolution_override: boolean;
  resolution_override: string;
  enable_random_circular_mask_shrink: boolean;
  enable_random_mask_rotate_crop: boolean;
}

interface ConceptTextConfig {
  prompt_source: string;
  prompt_path: string;
  enable_tag_shuffling: boolean;
  tag_delimiter: string;
  keep_tags_count: number;
  tag_dropout_enable: boolean;
  tag_dropout_mode: string;
  tag_dropout_probability: number;
}

interface Concept {
  id: string;
  name: string;
  path: string;
  seed: number;
  enabled: boolean;
  type: string;
  include_subdirectories: boolean;
  image_variations: number;
  text_variations: number;
  balancing: number;
  balancing_strategy: string;
  loss_weight: number;
  image: ConceptImageConfig;
  text: ConceptTextConfig;
}

const defaultImageConfig: ConceptImageConfig = {
  enable_crop_jitter: true,
  enable_random_flip: false,
  enable_fixed_flip: false,
  enable_random_rotate: false,
  enable_fixed_rotate: false,
  random_rotate_max_angle: 0.0,
  enable_random_brightness: false,
  enable_fixed_brightness: false,
  random_brightness_max_strength: 0.0,
  enable_random_contrast: false,
  enable_fixed_contrast: false,
  random_contrast_max_strength: 0.0,
  enable_random_saturation: false,
  enable_fixed_saturation: false,
  random_saturation_max_strength: 0.0,
  enable_random_hue: false,
  enable_fixed_hue: false,
  random_hue_max_strength: 0.0,
  enable_resolution_override: false,
  resolution_override: '512',
  enable_random_circular_mask_shrink: false,
  enable_random_mask_rotate_crop: false,
};

const defaultTextConfig: ConceptTextConfig = {
  prompt_source: 'sample',
  prompt_path: '',
  enable_tag_shuffling: false,
  tag_delimiter: ',',
  keep_tags_count: 1,
  tag_dropout_enable: false,
  tag_dropout_mode: 'FULL',
  tag_dropout_probability: 0.0,
};

const createDefaultConcept = (name: string = '', path: string = ''): Concept => ({
  id: `concept-${Date.now()}-${Math.random()}`,
  name,
  path,
  seed: Math.floor(Math.random() * 2147483647),
  enabled: true,
  type: 'STANDARD',
  include_subdirectories: false,
  image_variations: 1,
  text_variations: 1,
  balancing: 1.0,
  balancing_strategy: 'REPEATS',
  loss_weight: 1.0,
  image: { ...defaultImageConfig },
  text: { ...defaultTextConfig },
});

type DetailTab = 'general' | 'image' | 'text';

export function ConceptsView() {
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [showDisabled, setShowDisabled] = useState(true);
  const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null);
  const [detailTab, setDetailTab] = useState<DetailTab>('general');
  const [browsingFor, setBrowsingFor] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState('/home/alex/OneTrainer/training_concepts');
  const [browserEntries, setBrowserEntries] = useState<any[]>([]);
  const [browserLoading, setBrowserLoading] = useState(false);

  // Get config from store
  const { config, updateConfig } = useConfigStore();

  // Database status
  const { dbEnabled } = useDatabase();



  // Load concepts from database when DB mode is enabled
  useEffect(() => {
    if (!dbEnabled) return;

    const loadDbConcepts = async () => {
      try {
        const res = await databaseApi.listConcepts();
        const dbConcepts: Concept[] = res.data.map((dc: DbConcept) => ({
          id: `db-${dc.id}`,
          name: dc.name,
          path: dc.path,
          seed: dc.config?.seed || 0,
          enabled: dc.enabled,
          type: dc.concept_type,
          include_subdirectories: dc.config?.include_subdirectories || false,
          image_variations: dc.config?.image_variations || 1,
          text_variations: dc.config?.text_variations || 1,
          balancing: dc.config?.balancing || 1.0,
          balancing_strategy: dc.config?.balancing_strategy || 'REPEATS',
          loss_weight: dc.config?.loss_weight || 1.0,
          image: dc.config?.image ? { ...defaultImageConfig, ...dc.config.image } : { ...defaultImageConfig },
          text: dc.config?.text ? { ...defaultTextConfig, ...dc.config.text } : { ...defaultTextConfig },
        }));
        setConcepts(dbConcepts);
      } catch (err) {
        console.error('Failed to load concepts from database:', err);
      }
    };

    loadDbConcepts();
  }, [dbEnabled]);

  // Load concepts from Config with fallback to global file
  // Priority: 1. Config's embedded concepts → 2. Config's concept_file_name → 3. Global file fallback
  useEffect(() => {
    if (dbEnabled) return;

    const loadConfigConcepts = async () => {
      const c = config as any;

      // CASE 1: Config explicitly has concepts array (from preset) - USE IT
      if (c?.concepts && Array.isArray(c.concepts)) {
        const loadedConcepts: Concept[] = c.concepts.map((fc: any, index: number) => ({
          id: `concept-${index}-${Date.now()}`,
          name: fc.name || '',
          path: fc.path || '',
          seed: fc.seed || 0,
          enabled: fc.enabled !== false,
          type: fc.type || fc.concept_type || 'STANDARD',
          include_subdirectories: fc.include_subdirectories || false,
          image_variations: fc.image_variations || 1,
          text_variations: fc.text_variations || 1,
          balancing: fc.balancing || 1.0,
          balancing_strategy: fc.balancing_strategy || 'REPEATS',
          loss_weight: fc.loss_weight || 1.0,
          image: fc.image ? { ...defaultImageConfig, ...fc.image } : { ...defaultImageConfig },
          text: fc.text ? { ...defaultTextConfig, ...fc.text } : { ...defaultTextConfig },
        }));

        // Prevent infinite loop: compare with current state
        const currentNormalized = concepts.map(c => ({
          name: c.name, path: c.path, seed: c.seed, enabled: c.enabled, type: c.type,
          include_subdirectories: c.include_subdirectories, image_variations: c.image_variations,
          text_variations: c.text_variations, balancing: c.balancing, balancing_strategy: c.balancing_strategy,
          loss_weight: c.loss_weight, image: c.image, text: c.text
        }));

        const newNormalized = loadedConcepts.map(c => ({
          name: c.name, path: c.path, seed: c.seed, enabled: c.enabled, type: c.type,
          include_subdirectories: c.include_subdirectories, image_variations: c.image_variations,
          text_variations: c.text_variations, balancing: c.balancing, balancing_strategy: c.balancing_strategy,
          loss_weight: c.loss_weight, image: c.image, text: c.text
        }));

        if (JSON.stringify(currentNormalized) !== JSON.stringify(newNormalized)) {
          console.log('[Concepts] Loaded', loadedConcepts.length, 'concepts from preset/config');
          setConcepts(loadedConcepts);
        }
        return;
      }

      // CASE 2: Config has concept_file_name - load from that file
      if (c?.concept_file_name) {
        console.log('[Concepts] Loading from concept_file_name:', c.concept_file_name);
        try {
          const response = await configApi.getConceptsFile(c.concept_file_name);
          const fileConcepts = response.data.concepts;
          if (Array.isArray(fileConcepts)) {
            console.log('[Concepts] Loaded', fileConcepts.length, 'concepts from file');
            const loadedConcepts: Concept[] = fileConcepts.map((fc: any, index: number) => ({
              id: `concept-${index}-${Date.now()}`,
              name: fc.name || '',
              path: fc.path || '',
              seed: fc.seed || 0,
              enabled: fc.enabled !== false,
              type: fc.type || 'STANDARD',
              include_subdirectories: fc.include_subdirectories || false,
              image_variations: fc.image_variations || 1,
              text_variations: fc.text_variations || 1,
              balancing: fc.balancing || 1.0,
              balancing_strategy: fc.balancing_strategy || 'REPEATS',
              loss_weight: fc.loss_weight || 1.0,
              image: fc.image ? { ...defaultImageConfig, ...fc.image } : { ...defaultImageConfig },
              text: fc.text ? { ...defaultTextConfig, ...fc.text } : { ...defaultTextConfig },
            }));
            setConcepts(loadedConcepts);
          }
        } catch (err) {
          console.error('Failed to load concepts from file:', err);
        }
        return;
      }

      // CASE 3: Config has neither concepts nor concept_file_name - fall back to global file
      console.log('[Concepts] No concepts in config, falling back to global file');
      try {
        const response = await configApi.getConceptsFile('training_concepts/concepts.json');
        const fileConcepts = response.data.concepts;
        if (Array.isArray(fileConcepts) && fileConcepts.length > 0) {
          console.log('[Concepts] Loaded', fileConcepts.length, 'concepts from global file (fallback)');
          const loadedConcepts: Concept[] = fileConcepts.map((fc: any, index: number) => ({
            id: `concept-${index}-${Date.now()}`,
            name: fc.name || '',
            path: fc.path || '',
            seed: fc.seed || 0,
            enabled: fc.enabled !== false,
            type: fc.type || 'STANDARD',
            include_subdirectories: fc.include_subdirectories || false,
            image_variations: fc.image_variations || 1,
            text_variations: fc.text_variations || 1,
            balancing: fc.balancing || 1.0,
            balancing_strategy: fc.balancing_strategy || 'REPEATS',
            loss_weight: fc.loss_weight || 1.0,
            image: fc.image ? { ...defaultImageConfig, ...fc.image } : { ...defaultImageConfig },
            text: fc.text ? { ...defaultTextConfig, ...fc.text } : { ...defaultTextConfig },
          }));
          setConcepts(loadedConcepts);
        }
      } catch (err) {
        console.log('[Concepts] No global concepts file found');
      }
    };

    loadConfigConcepts();
  }, [config, dbEnabled]);

  // Track if we've loaded concepts from config at least once
  const [hasInitialized, setHasInitialized] = useState(false);

  // Sync concepts to config store and localStorage
  useEffect(() => {
    // Don't sync empty array back to config on initial load - wait for concepts to be loaded first
    if (!hasInitialized && concepts.length === 0) {
      return;
    }

    localStorage.setItem('onetrainer_concepts', JSON.stringify(concepts));

    // Create config-format concepts list
    const configConcepts = concepts.map(c => ({
      name: c.name,
      path: c.path,
      seed: c.seed,
      enabled: c.enabled,
      type: c.type,
      include_subdirectories: c.include_subdirectories,
      image_variations: c.image_variations,
      text_variations: c.text_variations,
      balancing: c.balancing,
      balancing_strategy: c.balancing_strategy,
      loss_weight: c.loss_weight,
      image: c.image,
      text: c.text,
    }));

    // Check if update is needed (prevent infinite loops)
    const currentConfig = (config as any)?.concepts;
    // Simple deep equality check
    if (JSON.stringify(currentConfig) !== JSON.stringify(configConcepts)) {
      updateConfig({ concepts: configConcepts } as any);
    }

    // Mark as initialized after first successful sync
    if (!hasInitialized && concepts.length > 0) {
      setHasInitialized(true);
    }
  }, [concepts, hasInitialized]);

  const handleAddConcept = () => {
    const newConcept = createDefaultConcept();
    setConcepts([...concepts, newConcept]);
    setSelectedConcept(newConcept);
  };

  const handleCloneConcept = (concept: Concept) => {
    const cloned = { ...concept, id: `concept-${Date.now()}`, seed: Math.floor(Math.random() * 2147483647) };
    setConcepts([...concepts, cloned]);
  };

  const handleRemoveConcept = async (id: string) => {
    if (dbEnabled && id.startsWith('db-')) {
      try {
        await databaseApi.deleteConcept(id);
      } catch (err) {
        console.error('Failed to delete concept from DB:', err);
        return;
      }
    }
    setConcepts(concepts.filter((c) => c.id !== id));
    if (selectedConcept?.id === id) setSelectedConcept(null);
  };

  const handleToggleEnabled = async (id: string) => {
    const concept = concepts.find(c => c.id === id);
    if (!concept) return;

    if (dbEnabled && id.startsWith('db-')) {
      try {
        // Optimistic update
        setConcepts(concepts.map((c) => c.id === id ? { ...c, enabled: !c.enabled } : c));

        // Use full update for now since toggle endpoint logic is internal to repo but exposed via update_config
        // Reconstruct config object for update
        // We only need to send what changed + required fields if strict, but update_config calculates diff
        // Let's send a minimal update or the full concept config? Repository update_config handles partials?
        // Looking at backend code, update_config accepts config_dict. 
        // It does: concept.enabled = config_dict.get('enabled', concept.enabled)
        // So we can send just { enabled: !enabled }
        await databaseApi.updateConcept(id, { enabled: !concept.enabled });
      } catch (err) {
        console.error('Failed to toggle concept in DB:', err);
        // Revert on failure
        setConcepts(concepts.map((c) => c.id === id ? { ...c, enabled: concept.enabled } : c));
      }
    } else {
      setConcepts(concepts.map((c) => c.id === id ? { ...c, enabled: !c.enabled } : c));
    }
  };

  const handleUpdateConcept = async (id: string, updates: Partial<Concept>) => {
    // Optimistic update
    setConcepts(concepts.map((c) => c.id === id ? { ...c, ...updates } : c));
    if (selectedConcept?.id === id) {
      setSelectedConcept({ ...selectedConcept, ...updates });
    }

    if (dbEnabled && id.startsWith('db-')) {
      try {
        // Debouncing would be ideal here for text inputs, but for now direct update
        // Prepare update dictionary. Flat structure expected by backend logic somewhat?
        // Backend repositories/concept_repository.py update_config takes nested?
        // It looks for keys: 'enabled', 'name', 'path', 'type', 'seed'... 
        // And 'image' dict, 'text' dict. 
        // So we need to structure the payload correctly.

        const current = concepts.find(c => c.id === id);
        if (!current) return;

        const updated = { ...current, ...updates };

        // Construct backend-friendly payload
        const payload: any = {
          name: updated.name,
          path: updated.path,
          type: updated.type,
          enabled: updated.enabled,
          seed: updated.seed,
          include_subdirectories: updated.include_subdirectories,
          image_variations: updated.image_variations,
          text_variations: updated.text_variations,
          balancing: updated.balancing,
          balancing_strategy: updated.balancing_strategy,
          loss_weight: updated.loss_weight,
          image: updated.image,
          text: updated.text
        };

        await databaseApi.updateConcept(id, payload);
      } catch (err) {
        console.error('Failed to update concept in DB:', err);
      }
    }
  };

  const filteredConcepts = concepts.filter((c) => {
    if (!showDisabled && !c.enabled) return false;
    if (typeFilter !== 'ALL' && c.type !== typeFilter) return false;
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      return c.name.toLowerCase().includes(term) || c.path.toLowerCase().includes(term);
    }
    return true;
  });

  const handleBrowse = async (conceptId: string) => {
    setBrowsingFor(conceptId);
    setBrowserLoading(true);
    try {
      const concept = concepts.find((c) => c.id === conceptId);
      const response = await filesystemApi.browse(concept?.path || currentPath);
      setBrowserEntries(response.data.entries);
      setCurrentPath(response.data.path);
    } catch (err) {
      console.error('Failed to browse:', err);
    } finally {
      setBrowserLoading(false);
    }
  };

  const navigateTo = async (path: string) => {
    setBrowserLoading(true);
    try {
      const response = await filesystemApi.browse(path);
      setBrowserEntries(response.data.entries);
      setCurrentPath(response.data.path);
    } catch (err) {
      console.error('Failed to navigate:', err);
    } finally {
      setBrowserLoading(false);
    }
  };

  const selectPath = (path: string) => {
    if (browsingFor) {
      handleUpdateConcept(browsingFor, { path });
      setBrowsingFor(null);
    }
  };

  const disabledCount = concepts.filter(c => !c.enabled).length;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white">Training Concepts</h1>
        <button onClick={handleAddConcept}
          className="bg-primary hover:bg-primary-hover text-white px-4 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Add Concept
        </button>
      </div>

      {/* Search & Filter Bar */}
      <div className="px-6 py-3 border-b border-dark-border bg-dark-surface flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <Search className="w-4 h-4 text-muted" />
          <input type="text" value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Filter..." className="input text-sm w-48" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">Type:</span>
          <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} className="input text-sm">
            <option value="ALL">ALL</option>
            <option value="STANDARD">STANDARD</option>
            <option value="VALIDATION">VALIDATION</option>
            <option value="PRIOR_PREDICTION">PRIOR_PREDICTION</option>
          </select>
        </div>
        <label className="flex items-center gap-2 text-sm text-white cursor-pointer">
          <input type="checkbox" checked={showDisabled} onChange={(e) => setShowDisabled(e.target.checked)} className="rounded" />
          Show Disabled {disabledCount > 0 && `(${disabledCount})`}
        </label>
        <button onClick={() => { setSearchTerm(''); setTypeFilter('ALL'); setShowDisabled(true); }}
          className="px-3 py-1 text-xs bg-dark-border hover:bg-dark-hover text-white rounded">Clear</button>
      </div>

      {/* Main Content - Grid + Detail Panel */}
      <div className="flex-1 flex overflow-hidden">
        {/* Concepts Grid */}
        <div className="flex-1 overflow-auto p-4">
          {filteredConcepts.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted">
              <ImageIcon className="w-16 h-16 mb-4 opacity-20" />
              <p className="text-sm">{concepts.length === 0 ? 'No concepts configured.' : 'No concepts match filters.'}</p>
              {concepts.length === 0 && (
                <button onClick={handleAddConcept} className="mt-4 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg text-sm">
                  Add First Concept
                </button>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
              {filteredConcepts.map((concept) => (
                <div key={concept.id}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('Clicking concept:', concept.id, concept.name);
                    // Ensure concept has all required nested properties
                    const conceptWithDefaults = {
                      ...concept,
                      image: concept.image ? { ...defaultImageConfig, ...concept.image } : { ...defaultImageConfig },
                      text: concept.text ? { ...defaultTextConfig, ...concept.text } : { ...defaultTextConfig },
                    };
                    setSelectedConcept(conceptWithDefaults);
                    setDetailTab('general');
                  }}
                  className={`relative bg-dark-card rounded-lg border cursor-pointer transition-all hover:border-primary
                    ${selectedConcept?.id === concept.id ? 'border-primary ring-1 ring-primary' : 'border-dark-border'}
                    ${!concept.enabled ? 'opacity-50' : ''}`}>
                  {/* Preview Image */}
                  <div className="aspect-square bg-dark-bg rounded-t-lg flex items-center justify-center overflow-hidden">
                    <ImageIcon className="w-12 h-12 text-dark-border" />
                  </div>

                  {/* Delete Button - Top Left */}
                  <div className="absolute top-1 left-1">
                    <button onClick={(e) => { e.stopPropagation(); handleRemoveConcept(concept.id); }}
                      className="w-6 h-6 bg-red-600 hover:bg-red-500 text-white rounded text-xs flex items-center justify-center"
                      title="Delete concept">
                      <X className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Clone Button - Bottom Left */}
                  <div className="absolute bottom-8 left-1">
                    <button onClick={(e) => { e.stopPropagation(); handleCloneConcept(concept); }}
                      className="w-6 h-6 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs flex items-center justify-center"
                      title="Clone concept">
                      <Copy className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Enable Toggle */}
                  <div className="absolute top-1 right-1">
                    <button onClick={(e) => { e.stopPropagation(); handleToggleEnabled(concept.id); }}
                      className={`w-9 h-5 rounded-full relative ${concept.enabled ? 'bg-green-600' : 'bg-gray-600'}`}>
                      <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${concept.enabled ? 'translate-x-4' : 'translate-x-0'}`} />
                    </button>
                  </div>

                  {/* Name + Edit Button */}
                  <div className="p-2 flex items-center justify-between gap-1">
                    <p className="text-xs text-white truncate flex-1 text-center">
                      {concept.name || concept.path.split('/').pop() || 'Untitled'}
                    </p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        console.log('Edit button clicked for:', concept.name);
                        const conceptWithDefaults = {
                          ...concept,
                          image: concept.image ? { ...defaultImageConfig, ...concept.image } : { ...defaultImageConfig },
                          text: concept.text ? { ...defaultTextConfig, ...concept.text } : { ...defaultTextConfig },
                        };
                        setSelectedConcept(conceptWithDefaults);
                        setDetailTab('general');
                      }}
                      className="w-6 h-6 bg-primary hover:bg-primary-hover text-white rounded text-xs flex items-center justify-center flex-shrink-0"
                      title="Edit concept"
                    >
                      ✎
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Detail Panel - Modal on small screens, sidebar on large */}
        {selectedConcept && (
          <>
            {/* Backdrop for mobile */}
            <div
              className="fixed inset-0 bg-black/50 z-40 lg:hidden"
              onClick={() => setSelectedConcept(null)}
            />
            {/* Panel */}
            <div className="fixed inset-y-0 right-0 w-full max-w-md z-50 lg:relative lg:z-auto lg:w-96 lg:min-w-[384px] border-l border-dark-border bg-dark-surface overflow-y-auto flex-shrink-0">
              <div className="p-4 border-b border-dark-border flex items-center justify-between">
                <h2 className="text-sm font-medium text-white">Concept Details</h2>
                <button onClick={() => setSelectedConcept(null)} className="text-muted hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Tabs */}
              <div className="border-b border-dark-border flex">
                {(['general', 'image', 'text'] as DetailTab[]).map(tab => (
                  <button key={tab} onClick={() => setDetailTab(tab)}
                    className={`px-4 py-2 text-sm capitalize ${detailTab === tab ? 'text-white border-b-2 border-primary' : 'text-muted hover:text-white'}`}>
                    {tab}
                  </button>
                ))}
              </div>

              <div className="p-4 space-y-3">
                {/* General Tab */}
                {detailTab === 'general' && (
                  <>
                    <div>
                      <label className="text-xs text-muted block mb-1">Name</label>
                      <input type="text" value={selectedConcept.name}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { name: e.target.value })}
                        className="input w-full text-sm" placeholder="Concept name" />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Path</label>
                      <div className="flex gap-2">
                        <input type="text" value={selectedConcept.path}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { path: e.target.value })}
                          className="input flex-1 text-sm" placeholder="/path/to/images" />
                        <button onClick={() => handleBrowse(selectedConcept.id)}
                          className="p-2 bg-dark-border hover:bg-dark-hover rounded">
                          <Folder className="w-4 h-4 text-muted" />
                        </button>
                      </div>
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Type</label>
                      <select value={selectedConcept.type}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { type: e.target.value })}
                        className="input w-full text-sm">
                        <option value="STANDARD">Standard</option>
                        <option value="VALIDATION">Validation</option>
                        <option value="PRIOR_PREDICTION">Prior Prediction</option>
                      </select>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-muted block mb-1">Balancing</label>
                        <input type="text" value={selectedConcept.balancing}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { balancing: parseFloat(e.target.value) || 1 })}
                          className="input w-full text-sm" step="0.1" min="0" />
                      </div>
                      <div>
                        <label className="text-xs text-muted block mb-1">Loss Weight</label>
                        <input type="text" value={selectedConcept.loss_weight}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { loss_weight: parseFloat(e.target.value) || 1 })}
                          className="input w-full text-sm" step="0.1" min="0" />
                      </div>
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Balancing Strategy</label>
                      <select value={selectedConcept.balancing_strategy}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { balancing_strategy: e.target.value })}
                        className="input w-full text-sm">
                        <option value="REPEATS">Repeats</option>
                        <option value="SAMPLES">Samples</option>
                      </select>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-muted block mb-1">Image Variations</label>
                        <input type="text" value={selectedConcept.image_variations}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image_variations: parseInt(e.target.value) || 1 })}
                          className="input w-full text-sm" min="1" />
                      </div>
                      <div>
                        <label className="text-xs text-muted block mb-1">Text Variations</label>
                        <input type="text" value={selectedConcept.text_variations}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { text_variations: parseInt(e.target.value) || 1 })}
                          className="input w-full text-sm" min="1" />
                      </div>
                    </div>
                    <div className="flex items-center gap-2 py-1">
                      <input type="checkbox" id="subdirs" checked={selectedConcept.include_subdirectories}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { include_subdirectories: e.target.checked })}
                        className="rounded" />
                      <label htmlFor="subdirs" className="text-sm text-white">Include Subdirectories</label>
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Seed</label>
                      <input type="text" value={selectedConcept.seed}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { seed: parseInt(e.target.value) || 0 })}
                        className="input w-full text-sm" />
                    </div>
                    <button onClick={() => handleRemoveConcept(selectedConcept.id)}
                      className="w-full py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded text-sm flex items-center justify-center gap-2 mt-4">
                      <Trash2 className="w-4 h-4" />
                      Delete Concept
                    </button>
                  </>
                )}

                {/* Image Tab */}
                {detailTab === 'image' && selectedConcept.image && (
                  <>
                    <h3 className="text-xs font-medium text-muted uppercase">Augmentation</h3>
                    <div className="flex items-center gap-2 py-1">
                      <input type="checkbox" checked={selectedConcept.image.enable_crop_jitter ?? false}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_crop_jitter: e.target.checked } })}
                        className="rounded" />
                      <span className="text-sm text-white">Crop Jitter</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex items-center gap-2 py-1">
                        <input type="checkbox" checked={selectedConcept.image.enable_random_flip}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_random_flip: e.target.checked } })}
                          className="rounded" />
                        <span className="text-sm text-white">Random Flip</span>
                      </div>
                      <div className="flex items-center gap-2 py-1">
                        <input type="checkbox" checked={selectedConcept.image.enable_fixed_flip}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_fixed_flip: e.target.checked } })}
                          className="rounded" />
                        <span className="text-sm text-white">Fixed Flip</span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex items-center gap-2 py-1">
                        <input type="checkbox" checked={selectedConcept.image.enable_random_rotate}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_random_rotate: e.target.checked } })}
                          className="rounded" />
                        <span className="text-sm text-white">Random Rotate</span>
                      </div>
                      <div>
                        <label className="text-xs text-muted block mb-1">Max Angle</label>
                        <input type="text" value={selectedConcept.image.random_rotate_max_angle}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, random_rotate_max_angle: parseFloat(e.target.value) || 0 } })}
                          className="input w-full text-sm" step="1" />
                      </div>
                    </div>
                    <h3 className="text-xs font-medium text-muted uppercase mt-3">Color</h3>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex items-center gap-2 py-1">
                        <input type="checkbox" checked={selectedConcept.image.enable_random_brightness}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_random_brightness: e.target.checked } })}
                          className="rounded" />
                        <span className="text-sm text-white">Brightness</span>
                      </div>
                      <div>
                        <label className="text-xs text-muted block mb-1">Strength</label>
                        <input type="text" value={selectedConcept.image.random_brightness_max_strength}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, random_brightness_max_strength: parseFloat(e.target.value) || 0 } })}
                          className="input w-full text-sm" step="0.1" />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex items-center gap-2 py-1">
                        <input type="checkbox" checked={selectedConcept.image.enable_random_contrast}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_random_contrast: e.target.checked } })}
                          className="rounded" />
                        <span className="text-sm text-white">Contrast</span>
                      </div>
                      <div>
                        <label className="text-xs text-muted block mb-1">Strength</label>
                        <input type="text" value={selectedConcept.image.random_contrast_max_strength}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, random_contrast_max_strength: parseFloat(e.target.value) || 0 } })}
                          className="input w-full text-sm" step="0.1" />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex items-center gap-2 py-1">
                        <input type="checkbox" checked={selectedConcept.image.enable_random_saturation}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_random_saturation: e.target.checked } })}
                          className="rounded" />
                        <span className="text-sm text-white">Saturation</span>
                      </div>
                      <div>
                        <label className="text-xs text-muted block mb-1">Strength</label>
                        <input type="text" value={selectedConcept.image.random_saturation_max_strength}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, random_saturation_max_strength: parseFloat(e.target.value) || 0 } })}
                          className="input w-full text-sm" step="0.1" />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex items-center gap-2 py-1">
                        <input type="checkbox" checked={selectedConcept.image.enable_random_hue}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_random_hue: e.target.checked } })}
                          className="rounded" />
                        <span className="text-sm text-white">Hue</span>
                      </div>
                      <div>
                        <label className="text-xs text-muted block mb-1">Strength</label>
                        <input type="text" value={selectedConcept.image.random_hue_max_strength}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, random_hue_max_strength: parseFloat(e.target.value) || 0 } })}
                          className="input w-full text-sm" step="0.1" />
                      </div>
                    </div>
                    <h3 className="text-xs font-medium text-muted uppercase mt-3">Resolution</h3>
                    <div className="flex items-center gap-2 py-1">
                      <input type="checkbox" checked={selectedConcept.image.enable_resolution_override}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_resolution_override: e.target.checked } })}
                        className="rounded" />
                      <span className="text-sm text-white">Override Resolution</span>
                    </div>
                    {selectedConcept.image.enable_resolution_override && (
                      <div>
                        <label className="text-xs text-muted block mb-1">Resolution</label>
                        <input type="text" value={selectedConcept.image.resolution_override}
                          onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, resolution_override: e.target.value } })}
                          className="input w-full text-sm" />
                      </div>
                    )}
                    <h3 className="text-xs font-medium text-muted uppercase mt-3">Mask</h3>
                    <div className="flex items-center gap-2 py-1">
                      <input type="checkbox" checked={selectedConcept.image.enable_random_circular_mask_shrink}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_random_circular_mask_shrink: e.target.checked } })}
                        className="rounded" />
                      <span className="text-sm text-white">Random Circular Mask Shrink</span>
                    </div>
                    <div className="flex items-center gap-2 py-1">
                      <input type="checkbox" checked={selectedConcept.image.enable_random_mask_rotate_crop}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { image: { ...selectedConcept.image, enable_random_mask_rotate_crop: e.target.checked } })}
                        className="rounded" />
                      <span className="text-sm text-white">Random Mask Rotate Crop</span>
                    </div>
                  </>
                )}

                {/* Text Tab */}
                {detailTab === 'text' && selectedConcept.text && (
                  <>
                    <div>
                      <label className="text-xs text-muted block mb-1">Prompt Source</label>
                      <select value={selectedConcept.text.prompt_source || 'sample'}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { text: { ...selectedConcept.text, prompt_source: e.target.value } })}
                        className="input w-full text-sm">
                        <option value="sample">From Sample (filename)</option>
                        <option value="txt">From .txt File</option>
                        <option value="filename">Filename</option>
                        <option value="concept">From Concept</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Prompt Path (optional)</label>
                      <input type="text" value={selectedConcept.text.prompt_path}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { text: { ...selectedConcept.text, prompt_path: e.target.value } })}
                        className="input w-full text-sm" placeholder="/path/to/prompts" />
                    </div>
                    <h3 className="text-xs font-medium text-muted uppercase mt-3">Tag Shuffling</h3>
                    <div className="flex items-center gap-2 py-1">
                      <input type="checkbox" checked={selectedConcept.text.enable_tag_shuffling}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { text: { ...selectedConcept.text, enable_tag_shuffling: e.target.checked } })}
                        className="rounded" />
                      <span className="text-sm text-white">Enable Tag Shuffling</span>
                    </div>
                    {selectedConcept.text.enable_tag_shuffling && (
                      <>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="text-xs text-muted block mb-1">Delimiter</label>
                            <input type="text" value={selectedConcept.text.tag_delimiter}
                              onChange={(e) => handleUpdateConcept(selectedConcept.id, { text: { ...selectedConcept.text, tag_delimiter: e.target.value } })}
                              className="input w-full text-sm" />
                          </div>
                          <div>
                            <label className="text-xs text-muted block mb-1">Keep Tags</label>
                            <input type="text" value={selectedConcept.text.keep_tags_count}
                              onChange={(e) => handleUpdateConcept(selectedConcept.id, { text: { ...selectedConcept.text, keep_tags_count: parseInt(e.target.value) || 1 } })}
                              className="input w-full text-sm" min="0" />
                          </div>
                        </div>
                      </>
                    )}
                    <h3 className="text-xs font-medium text-muted uppercase mt-3">Tag Dropout</h3>
                    <div className="flex items-center gap-2 py-1">
                      <input type="checkbox" checked={selectedConcept.text.tag_dropout_enable}
                        onChange={(e) => handleUpdateConcept(selectedConcept.id, { text: { ...selectedConcept.text, tag_dropout_enable: e.target.checked } })}
                        className="rounded" />
                      <span className="text-sm text-white">Enable Tag Dropout</span>
                    </div>
                    {selectedConcept.text.tag_dropout_enable && (
                      <>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="text-xs text-muted block mb-1">Mode</label>
                            <select value={selectedConcept.text.tag_dropout_mode}
                              onChange={(e) => handleUpdateConcept(selectedConcept.id, { text: { ...selectedConcept.text, tag_dropout_mode: e.target.value } })}
                              className="input w-full text-sm">
                              <option value="FULL">Full</option>
                              <option value="PARTIAL">Partial</option>
                            </select>
                          </div>
                          <div>
                            <label className="text-xs text-muted block mb-1">Probability</label>
                            <input type="text" value={selectedConcept.text.tag_dropout_probability}
                              onChange={(e) => handleUpdateConcept(selectedConcept.id, { text: { ...selectedConcept.text, tag_dropout_probability: parseFloat(e.target.value) || 0 } })}
                              className="input w-full text-sm" step="0.1" min="0" max="1" />
                          </div>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          </>
        )}

        {/* Directory Browser Modal */}
        {browsingFor && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
            <div className="bg-dark-surface border border-dark-border rounded-lg w-full max-w-2xl max-h-[80vh] flex flex-col">
              <div className="px-6 py-4 border-b border-dark-border flex items-center justify-between">
                <h2 className="text-lg font-medium text-white">Select Directory</h2>
                <button onClick={() => setBrowsingFor(null)} className="text-muted hover:text-white">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="px-6 py-3 border-b border-dark-border bg-dark-bg flex items-center gap-2">
                <button onClick={() => navigateTo(currentPath.split('/').slice(0, -1).join('/') || '/')}
                  disabled={browserLoading} className="px-3 py-1 text-sm bg-dark-surface border border-dark-border rounded hover:bg-dark-hover text-white disabled:opacity-50">
                  ↑ Up
                </button>
                <span className="text-sm text-muted truncate">{currentPath}</span>
              </div>

              <div className="flex-1 overflow-auto">
                {browserLoading && <div className="px-6 py-4 text-center text-muted">Loading...</div>}
                {!browserLoading && browserEntries.filter(e => e.is_directory).map((entry) => (
                  <div key={entry.name} onDoubleClick={() => navigateTo(`${currentPath}/${entry.name}`)}
                    className="px-6 py-2 hover:bg-dark-hover cursor-pointer flex items-center gap-2">
                    <Folder className="w-4 h-4 text-muted" />
                    <span className="text-white text-sm">{entry.name}</span>
                  </div>
                ))}
              </div>

              <div className="px-6 py-4 border-t border-dark-border flex justify-end gap-3">
                <button onClick={() => setBrowsingFor(null)} className="px-4 py-2 text-sm text-muted hover:text-white hover:bg-dark-hover rounded">Cancel</button>
                <button onClick={() => selectPath(currentPath)} className="px-4 py-2 text-sm bg-primary hover:bg-primary-hover text-white rounded">Select</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
