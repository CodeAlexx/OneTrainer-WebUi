import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { trainingWs } from './lib/api';

// Connect WebSocket for real-time training updates
trainingWs.connect();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
