import { useConfigStore } from '../../stores/configStore';

const CLOUD_TYPES = ['RUNPOD', 'LINUX'];
const FILE_SYNC_METHODS = ['NATIVE_SCP', 'FABRIC_SFTP'];
const CLOUD_ACTIONS = ['None', 'STOP', 'DELETE'];

export function CloudView() {
  const { config, updateConfig } = useConfigStore();

  // Cloud config from store (nested under 'cloud')
  const cloud = (config as any)?.cloud || {};
  const secrets = (config as any)?.cloud_secrets || {};

  // CloudConfig fields
  const enabled = cloud.enabled ?? false;
  const cloudType = cloud.type ?? 'RUNPOD';
  const fileSync = cloud.file_sync ?? 'NATIVE_SCP';
  const createCloud = cloud.create ?? true;
  const cloudName = cloud.name ?? 'OneTrainer';
  const tensorboardTunnel = cloud.tensorboard_tunnel ?? true;
  const subType = cloud.sub_type ?? '';
  const gpuType = cloud.gpu_type ?? '';
  const volumeSize = cloud.volume_size ?? 100;
  const minDownload = cloud.min_download ?? 0;
  const remoteDir = cloud.remote_dir ?? '/workspace';
  const huggingfaceCacheDir = cloud.huggingface_cache_dir ?? '/workspace/huggingface_cache';
  const onetrainerDir = cloud.onetrainer_dir ?? '/workspace/OneTrainer';
  const installOnetrainer = cloud.install_onetrainer ?? true;
  const updateOnetrainer = cloud.update_onetrainer ?? true;
  const installCmd = cloud.install_cmd ?? 'git clone https://github.com/Nerogar/OneTrainer';
  const detachTrainer = cloud.detach_trainer ?? false;
  const runId = cloud.run_id ?? 'job1';
  const downloadSamples = cloud.download_samples ?? true;
  const downloadOutputModel = cloud.download_output_model ?? true;
  const downloadSaves = cloud.download_saves ?? true;
  const downloadBackups = cloud.download_backups ?? false;
  const downloadTensorboard = cloud.download_tensorboard ?? false;
  const deleteWorkspace = cloud.delete_workspace ?? false;
  const onFinish = cloud.on_finish ?? 'None';
  const onError = cloud.on_error ?? 'None';
  const onDetachedFinish = cloud.on_detached_finish ?? 'None';
  const onDetachedError = cloud.on_detached_error ?? 'None';

  // CloudSecretsConfig fields
  const apiKey = secrets.api_key ?? '';
  const host = secrets.host ?? '';
  const port = secrets.port ?? 0;
  const user = secrets.user ?? 'root';
  const cloudId = secrets.id ?? '';
  const keyFile = secrets.key_file ?? '';
  const password = secrets.password ?? '';

  const handleCloudChange = (field: string, value: any) => {
    updateConfig({ cloud: { ...cloud, [field]: value } } as any);
  };

  const handleSecretsChange = (field: string, value: any) => {
    updateConfig({ cloud_secrets: { ...secrets, [field]: value } } as any);
  };

  const handleCreateViaWebsite = () => {
    if (cloudType === 'RUNPOD') {
      window.open('https://www.runpod.io/', '_blank');
    }
  };

  const handleReattachNow = async () => {
    try {
      await fetch('/api/cloud/reattach', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId })
      });
    } catch (err) {
      console.error('Failed to reattach:', err);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-dark-border bg-dark-surface">
        <h1 className="text-lg font-medium text-white">Cloud Training</h1>
        <button
          onClick={handleCreateViaWebsite}
          className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-1.5 rounded-lg text-sm font-medium"
        >
          Create cloud via website
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl space-y-6">

          {/* Enable Toggle */}
          <div className="flex items-center gap-4 bg-dark-surface rounded-lg border border-dark-border p-4">
            <label className="text-sm text-muted">Enabled</label>
            <button
              onClick={() => handleCloudChange('enabled', !enabled)}
              className={`w-9 h-5 rounded-full relative flex-shrink-0 ${enabled ? 'bg-cyan-600' : 'bg-gray-600'}`}
            >
              <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${enabled ? 'translate-x-4' : 'translate-x-0'}`} />
            </button>
          </div>

          {/* Main Settings Grid */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Cloud Configuration</h2>
            </div>
            <div className="p-4 grid grid-cols-3 gap-4">
              {/* Type */}
              <div>
                <label className="text-xs text-muted block mb-1">Type</label>
                <select
                  value={cloudType}
                  onChange={(e) => handleCloudChange('type', e.target.value)}
                  className="input w-full text-sm bg-cyan-600 text-white"
                >
                  {CLOUD_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              {/* Remote Directory */}
              <div>
                <label className="text-xs text-muted block mb-1">Remote Directory</label>
                <input
                  type="text"
                  value={remoteDir}
                  onChange={(e) => handleCloudChange('remote_dir', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Create cloud via API */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Create cloud via API</label>
                <button
                  onClick={() => handleCloudChange('create', !createCloud)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${createCloud ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${createCloud ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* File sync method */}
              <div>
                <label className="text-xs text-muted block mb-1">File sync method</label>
                <select
                  value={fileSync}
                  onChange={(e) => handleCloudChange('file_sync', e.target.value)}
                  className="input w-full text-sm bg-cyan-600 text-white"
                >
                  {FILE_SYNC_METHODS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              {/* OneTrainer Directory */}
              <div>
                <label className="text-xs text-muted block mb-1">OneTrainer Directory</label>
                <input
                  type="text"
                  value={onetrainerDir}
                  onChange={(e) => handleCloudChange('onetrainer_dir', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Cloud name */}
              <div>
                <label className="text-xs text-muted block mb-1">Cloud name</label>
                <input
                  type="text"
                  value={cloudName}
                  onChange={(e) => handleCloudChange('name', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* API key */}
              <div>
                <label className="text-xs text-muted block mb-1">API key</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => handleSecretsChange('api_key', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Huggingface cache Directory */}
              <div>
                <label className="text-xs text-muted block mb-1">Huggingface cache Directory</label>
                <input
                  type="text"
                  value={huggingfaceCacheDir}
                  onChange={(e) => handleCloudChange('huggingface_cache_dir', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Sub Type */}
              <div>
                <label className="text-xs text-muted block mb-1">Type (sub)</label>
                <input
                  type="text"
                  value={subType}
                  onChange={(e) => handleCloudChange('sub_type', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Hostname */}
              <div>
                <label className="text-xs text-muted block mb-1">Hostname</label>
                <input
                  type="text"
                  value={host}
                  onChange={(e) => handleSecretsChange('host', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Install OneTrainer */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Install OneTrainer</label>
                <button
                  onClick={() => handleCloudChange('install_onetrainer', !installOnetrainer)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${installOnetrainer ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${installOnetrainer ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* GPU */}
              <div>
                <label className="text-xs text-muted block mb-1">GPU</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={gpuType}
                    onChange={(e) => handleCloudChange('gpu_type', e.target.value)}
                    className="input flex-1 text-sm"
                  />
                  <button className="px-2 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-sm">...</button>
                </div>
              </div>

              {/* Port */}
              <div>
                <label className="text-xs text-muted block mb-1">Port</label>
                <input
                  type="text"
                  value={port}
                  onChange={(e) => handleSecretsChange('port', parseInt(e.target.value) || 0)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Install command */}
              <div className="col-span-2">
                <label className="text-xs text-muted block mb-1">Install command</label>
                <input
                  type="text"
                  value={installCmd}
                  onChange={(e) => handleCloudChange('install_cmd', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Volume size */}
              <div>
                <label className="text-xs text-muted block mb-1">Volume size</label>
                <input
                  type="text"
                  value={volumeSize}
                  onChange={(e) => handleCloudChange('volume_size', parseInt(e.target.value) || 100)}
                  className="input w-full text-sm"
                />
              </div>

              {/* User */}
              <div>
                <label className="text-xs text-muted block mb-1">User</label>
                <input
                  type="text"
                  value={user}
                  onChange={(e) => handleSecretsChange('user', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Update OneTrainer */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Update OneTrainer</label>
                <button
                  onClick={() => handleCloudChange('update_onetrainer', !updateOnetrainer)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${updateOnetrainer ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${updateOnetrainer ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Min download */}
              <div>
                <label className="text-xs text-muted block mb-1">Min download</label>
                <input
                  type="text"
                  value={minDownload}
                  onChange={(e) => handleCloudChange('min_download', parseInt(e.target.value) || 0)}
                  className="input w-full text-sm"
                />
              </div>

              {/* SSH keyfile path */}
              <div>
                <label className="text-xs text-muted block mb-1">SSH keyfile path</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={keyFile}
                    onChange={(e) => handleSecretsChange('key_file', e.target.value)}
                    className="input flex-1 text-sm"
                  />
                  <button className="px-2 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-sm">...</button>
                </div>
              </div>

              {/* SSH password */}
              <div>
                <label className="text-xs text-muted block mb-1">SSH password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => handleSecretsChange('password', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Detach remote trainer */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Detach remote trainer</label>
                <button
                  onClick={() => handleCloudChange('detach_trainer', !detachTrainer)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${detachTrainer ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${detachTrainer ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Action on finish */}
              <div>
                <label className="text-xs text-muted block mb-1">Action on finish</label>
                <select
                  value={onFinish}
                  onChange={(e) => handleCloudChange('on_finish', e.target.value)}
                  className="input w-full text-sm bg-cyan-600 text-white"
                >
                  {CLOUD_ACTIONS.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>

              {/* Cloud Id */}
              <div>
                <label className="text-xs text-muted block mb-1">Cloud Id</label>
                <input
                  type="text"
                  value={cloudId}
                  onChange={(e) => handleSecretsChange('id', e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              {/* Reattach Id */}
              <div>
                <label className="text-xs text-muted block mb-1">Reattach Id</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={runId}
                    onChange={(e) => handleCloudChange('run_id', e.target.value)}
                    className="input flex-1 text-sm"
                  />
                  <button
                    onClick={handleReattachNow}
                    className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-sm"
                  >
                    Reattach now
                  </button>
                </div>
              </div>

              {/* Action on error */}
              <div>
                <label className="text-xs text-muted block mb-1">Action on error</label>
                <select
                  value={onError}
                  onChange={(e) => handleCloudChange('on_error', e.target.value)}
                  className="input w-full text-sm bg-cyan-600 text-white"
                >
                  {CLOUD_ACTIONS.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>

              {/* Tensorboard TCP tunnel */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Tensorboard TCP tunnel</label>
                <button
                  onClick={() => handleCloudChange('tensorboard_tunnel', !tensorboardTunnel)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${tensorboardTunnel ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${tensorboardTunnel ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Action on detached finish */}
              <div>
                <label className="text-xs text-muted block mb-1">Action on detached finish</label>
                <select
                  value={onDetachedFinish}
                  onChange={(e) => handleCloudChange('on_detached_finish', e.target.value)}
                  className="input w-full text-sm bg-cyan-600 text-white"
                >
                  {CLOUD_ACTIONS.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>

              {/* Action on detached error */}
              <div>
                <label className="text-xs text-muted block mb-1">Action on detached error</label>
                <select
                  value={onDetachedError}
                  onChange={(e) => handleCloudChange('on_detached_error', e.target.value)}
                  className="input w-full text-sm bg-cyan-600 text-white"
                >
                  {CLOUD_ACTIONS.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* Download Options */}
          <div className="bg-dark-surface rounded-lg border border-dark-border">
            <div className="px-4 py-3 border-b border-dark-border">
              <h2 className="text-sm font-medium text-muted uppercase tracking-wider">Download Options</h2>
            </div>
            <div className="p-4 grid grid-cols-2 gap-4">
              {/* Download samples */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Download samples</label>
                <button
                  onClick={() => handleCloudChange('download_samples', !downloadSamples)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${downloadSamples ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${downloadSamples ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Download output model */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Download output model</label>
                <button
                  onClick={() => handleCloudChange('download_output_model', !downloadOutputModel)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${downloadOutputModel ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${downloadOutputModel ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Download saved checkpoints */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Download saved checkpoints</label>
                <button
                  onClick={() => handleCloudChange('download_saves', !downloadSaves)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${downloadSaves ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${downloadSaves ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Download backups */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Download backups</label>
                <button
                  onClick={() => handleCloudChange('download_backups', !downloadBackups)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${downloadBackups ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${downloadBackups ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Download tensorboard logs */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Download tensorboard logs</label>
                <button
                  onClick={() => handleCloudChange('download_tensorboard', !downloadTensorboard)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${downloadTensorboard ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${downloadTensorboard ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* Delete remote workspace */}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted">Delete remote workspace</label>
                <button
                  onClick={() => handleCloudChange('delete_workspace', !deleteWorkspace)}
                  className={`w-9 h-5 rounded-full relative flex-shrink-0 ${deleteWorkspace ? 'bg-cyan-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${deleteWorkspace ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
