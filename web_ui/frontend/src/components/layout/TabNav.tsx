import { Tabs, TabsList, TabsTrigger } from '../ui/tabs';

interface TabNavProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const tabs = [
  { id: 'general', label: 'General' },
  { id: 'model', label: 'Model' },
  { id: 'data', label: 'Data' },
  { id: 'concepts', label: 'Concepts' },
  { id: 'training', label: 'Training' },
  { id: 'sampling', label: 'Sampling' },
  { id: 'backup', label: 'Backup' },
  { id: 'tools', label: 'Tools' },
  { id: 'cloud', label: 'Cloud' },
  { id: 'lora', label: 'LoRA' },
];

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <div className="border-b border-dark-border bg-dark-surface px-6">
      <Tabs value={activeTab} onValueChange={onTabChange}>
        <TabsList className="w-full justify-start">
          {tabs.map((tab) => (
            <TabsTrigger key={tab.id} value={tab.id}>
              {tab.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
    </div>
  );
}
