import { useEffect, useState } from 'react';
import {
  LayoutDashboard,
  ListOrdered,
  Database,
  HardDrive,
  Settings,
  Sliders,
  Layers,
  Zap,
  Image,
  Wrench,
  Type,
  Cloud,
  Play,
  RefreshCw,
  BarChart3,
  Save,
  Box
} from 'lucide-react';
import { trainingWs } from '../../lib/api';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'queue', label: 'Training Queue', icon: ListOrdered },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'general', label: 'Configuration', icon: Sliders },
  { id: 'concepts', label: 'Concepts', icon: Layers },
  { id: 'training', label: 'Training', icon: Zap },
  { id: 'sampling', label: 'Sampling', icon: Image },
  { id: 'backup', label: 'Backup', icon: Save },
  { id: 'inference', label: 'Inference', icon: Play },
  { id: 'tensorboard', label: 'TensorBoard', icon: BarChart3 },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'embeddings', label: 'Embeddings', icon: Type },
  { id: 'cloud', label: 'Cloud', icon: Cloud },
  { id: 'database', label: 'Database', icon: HardDrive },
  { id: 'models', label: 'Models', icon: Box },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export function Sidebar({ activeView, onViewChange }: SidebarProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);

  useEffect(() => {
    const handleConnected = () => {
      setIsConnected(true);
      setIsReconnecting(false);
    };
    const handleDisconnected = () => setIsConnected(false);

    trainingWs.on('connected', handleConnected);
    trainingWs.on('disconnected', handleDisconnected);

    // Check initial state
    if (trainingWs['ws']?.readyState === WebSocket.OPEN) {
      setIsConnected(true);
    }

    return () => {
      trainingWs.off('connected', handleConnected);
      trainingWs.off('disconnected', handleDisconnected);
    };
  }, []);

  const handleReconnect = () => {
    setIsReconnecting(true);
    trainingWs.reconnect();
  };

  return (
    <aside className="w-56 bg-dark-surface border-r border-dark-border flex flex-col">
      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`
                w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                transition-colors duration-150
                ${isActive
                  ? 'border-l-2 border-white text-white bg-dark-hover/50'
                  : 'text-muted hover:text-white hover:bg-dark-hover'
                }
              `}
            >
              <Icon className="w-5 h-5" />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-dark-border space-y-2">
        {/* Connection Status */}
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
              {isReconnecting ? 'Reconnecting...' : isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <button
            onClick={handleReconnect}
            disabled={isReconnecting}
            className="p-1 hover:bg-dark-hover rounded text-muted hover:text-white disabled:opacity-50"
            title="Reconnect"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isReconnecting ? 'animate-spin' : ''}`} />
          </button>
        </div>
        {/* Stop Server Button */}
        <button
          onClick={() => {
            if (confirm('Stop the web UI server? You will need to restart it manually.')) {
              fetch('/api/system/shutdown', { method: 'POST' })
                .then(() => {
                  // Server is shutting down
                  window.close();
                })
                .catch(() => {
                  alert('Server may already be stopped');
                });
            }
          }}
          className="w-full flex items-center justify-center gap-2 px-3 py-1.5 rounded text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 border border-red-500/30 hover:border-red-500/50 transition-colors"
        >
          Stop Server
        </button>
      </div>
    </aside>
  );
}
