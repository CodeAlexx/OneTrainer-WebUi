import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { TrainingConfig, TrainingStatus } from '../types/config';

interface ConfigStore {
  config: Partial<TrainingConfig>;
  status: TrainingStatus;
  currentPreset: string | null;
  activeTab: string;
  workspacePath: string;

  setConfig: (config: Partial<TrainingConfig>) => void;
  updateConfig: (updates: Partial<TrainingConfig>) => void;
  setStatus: (status: TrainingStatus) => void;
  setCurrentPreset: (preset: string | null) => void;
  setActiveTab: (tab: string) => void;
  setWorkspacePath: (path: string) => void;
}

export const useConfigStore = create<ConfigStore>()(
  persist(
    (set) => ({
      config: {},
      status: {
        status: 'idle',
      },
      currentPreset: null,
      activeTab: 'general',
      workspacePath: '/home/alex/a_liza_zimage',

      setConfig: (config) => set({ config }),
      updateConfig: (updates) =>
        set((state) => ({ config: { ...state.config, ...updates } })),
      setStatus: (status) => set({ status }),
      setCurrentPreset: (preset) => set({ currentPreset: preset }),
      setActiveTab: (tab) => set({ activeTab: tab }),
      setWorkspacePath: (path) => set({ workspacePath: path }),
    }),
    {
      name: 'onetrainer-config',
      partialize: (state) => ({
        config: state.config,
        currentPreset: state.currentPreset,
        activeTab: state.activeTab,
        workspacePath: state.workspacePath,
      }),
    }
  )
);
