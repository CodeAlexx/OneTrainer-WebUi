import { create } from 'zustand';

const useTrainingStore = create((set) => ({
  // Training state
  status: 'idle', // 'idle' | 'running' | 'paused' | 'stopped'
  epoch: 0,
  step: 0,
  totalEpochs: 0,
  totalSteps: 0,
  loss: null,
  learningRate: null,
  eta: null,

  // Loss history for charting
  lossHistory: [],

  // Recent samples
  recentSamples: [],

  // WebSocket connection status
  wsConnected: false,

  // Actions
  setStatus: (status) => set({ status }),

  setProgress: (progress) => set({
    epoch: progress.epoch,
    step: progress.step,
    totalEpochs: progress.total_epochs,
    totalSteps: progress.total_steps,
    loss: progress.loss,
    learningRate: progress.learning_rate,
    eta: progress.eta,
  }),

  addLossPoint: (point) => set((state) => ({
    lossHistory: [...state.lossHistory, point].slice(-1000), // Keep last 1000 points
  })),

  clearLossHistory: () => set({ lossHistory: [] }),

  addSample: (sample) => set((state) => ({
    recentSamples: [sample, ...state.recentSamples].slice(0, 10), // Keep last 10 samples
  })),

  setWsConnected: (connected) => set({ wsConnected: connected }),

  reset: () => set({
    status: 'idle',
    epoch: 0,
    step: 0,
    totalEpochs: 0,
    totalSteps: 0,
    loss: null,
    learningRate: null,
    eta: null,
    lossHistory: [],
  }),
}));

export default useTrainingStore;
