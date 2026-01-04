import { Outlet, NavLink } from 'react-router-dom';
import { Activity, Settings, Image, Wifi, WifiOff } from 'lucide-react';
import { useEffect } from 'react';
import useTrainingStore from '../stores/trainingStore';
import apiClient from '../api/client';

function Layout() {
  const wsConnected = useTrainingStore((state) => state.wsConnected);

  useEffect(() => {
    // Connect WebSocket on mount
    apiClient.connectWebSocket();

    // Cleanup on unmount
    return () => {
      apiClient.disconnectWebSocket();
    };
  }, []);

  const navItems = [
    { path: '/training', label: 'Training', icon: Activity },
    { path: '/config', label: 'Config', icon: Settings },
    { path: '/samples', label: 'Samples', icon: Image },
  ];

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-dark-surface border-r border-dark-border flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-dark-border">
          <h1 className="text-2xl font-bold text-primary">OneTrainer</h1>
          <p className="text-sm text-gray-400 mt-1">Web UI</p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <li key={item.path}>
                  <NavLink
                    to={item.path}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                        isActive
                          ? 'bg-primary text-white'
                          : 'text-gray-300 hover:bg-dark-hover'
                      }`
                    }
                  >
                    <Icon size={20} />
                    <span className="font-medium">{item.label}</span>
                  </NavLink>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Connection status */}
        <div className="p-4 border-t border-dark-border">
          <div className="flex items-center gap-2 text-sm">
            {wsConnected ? (
              <>
                <Wifi size={16} className="text-success" />
                <span className="text-gray-400">Connected</span>
              </>
            ) : (
              <>
                <WifiOff size={16} className="text-danger" />
                <span className="text-gray-400">Disconnected</span>
              </>
            )}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}

export default Layout;
