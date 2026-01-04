import { useState, useEffect } from 'react';
import { Plus, X } from 'lucide-react';
import { useConfigStore } from '../../stores/configStore';

const TIME_UNITS = ['NEVER', 'EPOCH', 'STEP', 'SECOND', 'MINUTE', 'HOUR'];

interface Embedding {
  uuid: string;
  model_name: string;  // base embedding path
  placeholder: string;
  train: boolean;
  stop_training_after: number | null;
  stop_training_after_unit: string;
  token_count: number | null;
  initial_embedding_text: string;
  is_output_embedding: boolean;
}

const createDefaultEmbedding = (): Embedding => ({
  uuid: `emb-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  model_name: '',
  placeholder: '<embedding>',
  train: true,
  stop_training_after: null,
  stop_training_after_unit: 'NEVER',
  token_count: 1,
  initial_embedding_text: '*',
  is_output_embedding: false,
});

export function EmbeddingsView() {
  const { config, updateConfig } = useConfigStore();
  const [embeddings, setEmbeddings] = useState<Embedding[]>([]);

  // Load embeddings from config
  useEffect(() => {
    const c = config as any;
    if (c?.embeddings && Array.isArray(c.embeddings)) {
      setEmbeddings(c.embeddings.map((e: any) => ({
        uuid: e.uuid || `emb-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        model_name: e.model_name || '',
        placeholder: e.placeholder || '<embedding>',
        train: e.train !== false,
        stop_training_after: e.stop_training_after ?? null,
        stop_training_after_unit: e.stop_training_after_unit || 'NEVER',
        token_count: e.token_count ?? 1,
        initial_embedding_text: e.initial_embedding_text || '*',
        is_output_embedding: e.is_output_embedding || false,
      })));
    }
  }, [config]);

  // Save embeddings to config when changed
  const saveEmbeddings = (newEmbeddings: Embedding[]) => {
    setEmbeddings(newEmbeddings);
    updateConfig({ embeddings: newEmbeddings } as any);
  };

  const addEmbedding = () => {
    saveEmbeddings([...embeddings, createDefaultEmbedding()]);
  };

  const removeEmbedding = (uuid: string) => {
    saveEmbeddings(embeddings.filter(e => e.uuid !== uuid));
  };

  const updateEmbedding = (uuid: string, updates: Partial<Embedding>) => {
    saveEmbeddings(embeddings.map(e => e.uuid === uuid ? { ...e, ...updates } : e));
  };

  const disableAll = () => {
    saveEmbeddings(embeddings.map(e => ({ ...e, train: false })));
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white">Additional Embeddings</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={addEmbedding}
            className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-1.5 rounded-lg text-sm font-medium"
          >
            add embedding
          </button>
          <button
            onClick={disableAll}
            className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-1.5 rounded-lg text-sm font-medium"
          >
            Disable
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="space-y-4">
          {embeddings.length === 0 ? (
            <div className="text-center text-muted py-12 bg-dark-surface rounded-lg border border-dark-border">
              No additional embeddings configured. Click "add embedding" to create one.
            </div>
          ) : (
            embeddings.map((embedding) => (
              <div key={embedding.uuid} className="bg-dark-surface rounded-lg border border-dark-border p-4">
                {/* Row 1: Controls + Base Embedding + Placeholder + Token Count */}
                <div className="flex items-center gap-4 mb-3">
                  {/* Delete/Add buttons */}
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => removeEmbedding(embedding.uuid)}
                      className="w-6 h-6 bg-red-600 hover:bg-red-500 text-white rounded text-xs flex items-center justify-center"
                    >
                      <X className="w-3 h-3" />
                    </button>
                    <button
                      onClick={addEmbedding}
                      className="w-6 h-6 bg-green-600 hover:bg-green-500 text-white rounded text-xs flex items-center justify-center"
                    >
                      <Plus className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Base Embedding */}
                  <div className="flex items-center gap-2 flex-1">
                    <label className="text-sm text-muted whitespace-nowrap">base embedding:</label>
                    <input
                      type="text"
                      value={embedding.model_name}
                      onChange={(e) => updateEmbedding(embedding.uuid, { model_name: e.target.value })}
                      className="input flex-1 text-sm"
                      placeholder="Path to base embedding file..."
                    />
                    <button className="px-2 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-sm">
                      ...
                    </button>
                  </div>

                  {/* Placeholder */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-muted">placeholder:</label>
                    <input
                      type="text"
                      value={embedding.placeholder}
                      onChange={(e) => updateEmbedding(embedding.uuid, { placeholder: e.target.value })}
                      className="input w-32 text-sm bg-cyan-900/30"
                    />
                  </div>

                  {/* Token Count */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-muted">token count:</label>
                    <input
                      type="text"
                      value={embedding.token_count ?? ''}
                      onChange={(e) => updateEmbedding(embedding.uuid, { token_count: parseInt(e.target.value) || null })}
                      className="input w-16 text-sm"
                    />
                  </div>
                </div>

                {/* Row 2: Toggles and Stop Training */}
                <div className="flex items-center gap-6">
                  {/* Train Toggle */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-muted">train:</label>
                    <button
                      onClick={() => updateEmbedding(embedding.uuid, { train: !embedding.train })}
                      className={`w-9 h-5 rounded-full relative flex-shrink-0 ${embedding.train ? 'bg-cyan-600' : 'bg-gray-600'}`}
                    >
                      <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${embedding.train ? 'translate-x-4' : 'translate-x-0'}`} />
                    </button>
                  </div>

                  {/* Output Embedding Toggle */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-muted">output embedding:</label>
                    <button
                      onClick={() => updateEmbedding(embedding.uuid, { is_output_embedding: !embedding.is_output_embedding })}
                      className={`w-9 h-5 rounded-full relative flex-shrink-0 ${embedding.is_output_embedding ? 'bg-cyan-600' : 'bg-gray-600'}`}
                    >
                      <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${embedding.is_output_embedding ? 'translate-x-4' : 'translate-x-0'}`} />
                    </button>
                  </div>

                  {/* Stop Training After */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-muted">stop training after:</label>
                    <input
                      type="text"
                      value={embedding.stop_training_after ?? ''}
                      onChange={(e) => updateEmbedding(embedding.uuid, { stop_training_after: parseInt(e.target.value) || null })}
                      className="input w-16 text-sm"
                    />
                    <select
                      value={embedding.stop_training_after_unit}
                      onChange={(e) => updateEmbedding(embedding.uuid, { stop_training_after_unit: e.target.value })}
                      className="input text-sm bg-cyan-600 text-white"
                    >
                      {TIME_UNITS.map(unit => (
                        <option key={unit} value={unit}>{unit}</option>
                      ))}
                    </select>
                  </div>

                  {/* Initial Embedding Text */}
                  <div className="flex items-center gap-2 flex-1">
                    <label className="text-sm text-muted">initial embedding text:</label>
                    <input
                      type="text"
                      value={embedding.initial_embedding_text}
                      onChange={(e) => updateEmbedding(embedding.uuid, { initial_embedding_text: e.target.value })}
                      className="input flex-1 text-sm"
                    />
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
