import { useState } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { DashboardView } from './components/views/DashboardView';
import { QueueView } from './components/views/QueueView';
import { DatasetsView } from './components/views/DatasetsView';
import { SettingsView } from './components/views/SettingsView';
import { GeneralView } from './components/views/GeneralView';
import { ConceptsView } from './components/views/ConceptsView';
import { TrainingView } from './components/views/TrainingView';
import { SamplingView } from './components/views/SamplingView';
import { ToolsView } from './components/views/ToolsView';
import { EmbeddingsView } from './components/views/EmbeddingsView';
import { CloudView } from './components/views/CloudView';
import { InferenceView } from './components/views/InferenceView';
import { TensorBoardView } from './components/views/TensorBoardView';
import { LoRAView } from './components/views/LoRAView';
import { BackupView } from './components/views/BackupView';
import { DatabaseView } from './components/views/DatabaseView';
import { ModelsSettingsView } from './components/views/ModelsSettingsView';

function App() {
  const [activeView, setActiveView] = useState('dashboard');

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <DashboardView />;
      case 'queue':
        return <QueueView />;
      case 'datasets':
        return <DatasetsView />;
      case 'settings':
        return <SettingsView />;
      case 'general':
        return <GeneralView />;
      case 'concepts':
        return <ConceptsView />;
      case 'training':
        return <TrainingView />;
      case 'lora':
        return <LoRAView />;
      case 'sampling':
        return <SamplingView />;
      case 'tools':
        return <ToolsView />;
      case 'embeddings':
        return <EmbeddingsView />;
      case 'cloud':
        return <CloudView />;
      case 'inference':
        return <InferenceView />;
      case 'tensorboard':
        return <TensorBoardView />;
      case 'backup':
        return <BackupView />;
      case 'database':
        return <DatabaseView />;
      case 'models':
        return <ModelsSettingsView />;
      default:
        return <DashboardView />;
    }
  };

  return (
    <div className="h-screen flex flex-col bg-dark-bg">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar activeView={activeView} onViewChange={setActiveView} />
        <main className="flex-1 overflow-auto">
          {renderView()}
        </main>
      </div>
    </div>
  );
}

export default App;
