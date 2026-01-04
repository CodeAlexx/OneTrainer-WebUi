import { useConfigStore } from '../../stores/configStore';
import { Play, Pause, Square } from 'lucide-react';
import { Button } from '../ui/button';

export function StatusBar() {
  const status = useConfigStore((state) => state.status);
  
  const getProgressPercentage = () => {
    if (!status.total_steps || !status.current_step) return 0;
    return (status.current_step / status.total_steps) * 100;
  };
  
  return (
    <footer className="h-16 border-t border-dark-border bg-dark-surface px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Button 
            variant="success" 
            size="sm"
            disabled={status.status === 'running'}
          >
            <Play className="h-4 w-4 mr-2" />
            Start Training
          </Button>
          <Button 
            variant="secondary" 
            size="sm"
            disabled={status.status !== 'running'}
          >
            <Pause className="h-4 w-4 mr-2" />
            Pause
          </Button>
          <Button 
            variant="danger" 
            size="sm"
            disabled={status.status === 'idle'}
          >
            <Square className="h-4 w-4 mr-2" />
            Stop
          </Button>
        </div>
        
        {status.status !== 'idle' && (
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <span>
              Epoch: {status.current_epoch || 0}/{status.total_epochs || 0}
            </span>
            <span>
              Step: {status.current_step || 0}/{status.total_steps || 0}
            </span>
            {status.loss !== undefined && (
              <span>Loss: {status.loss.toFixed(4)}</span>
            )}
            {status.samples_per_second !== undefined && (
              <span>{status.samples_per_second.toFixed(2)} it/s</span>
            )}
          </div>
        )}
      </div>
      
      {status.status === 'running' && (
        <div className="flex-1 max-w-md mx-8">
          <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
            <div 
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${getProgressPercentage()}%` }}
            />
          </div>
        </div>
      )}
      
      <div className="text-sm text-gray-400">
        Status: <span className={`font-medium ${
          status.status === 'running' ? 'text-success' :
          status.status === 'error' ? 'text-danger' :
          'text-gray-300'
        }`}>
          {status.status.charAt(0).toUpperCase() + status.status.slice(1)}
        </span>
      </div>
    </footer>
  );
}
