import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Label } from '../ui/label';
import { Input } from '../ui/input';
import { Switch } from '../ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';

export function TrainingTab() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Training Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="learning-rate">Learning Rate</Label>
              <Input
                id="learning-rate"
                type="text"
                placeholder="1e-4"
                step="0.0001"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="batch-size">Batch Size</Label>
              <Input
                id="batch-size"
                type="text"
                placeholder="1"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="epochs">Epochs</Label>
              <Input
                id="epochs"
                type="text"
                placeholder="100"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="gradient-steps">Gradient Accumulation Steps</Label>
              <Input
                id="gradient-steps"
                type="text"
                placeholder="1"
              />
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Optimizer Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="optimizer">Optimizer</Label>
            <Select defaultValue="adamw">
              <SelectTrigger id="optimizer">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="adam">Adam</SelectItem>
                <SelectItem value="adamw">AdamW</SelectItem>
                <SelectItem value="sgd">SGD</SelectItem>
                <SelectItem value="adafactor">Adafactor</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="scheduler">Learning Rate Scheduler</Label>
            <Select defaultValue="constant">
              <SelectTrigger id="scheduler">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="constant">Constant</SelectItem>
                <SelectItem value="linear">Linear</SelectItem>
                <SelectItem value="cosine">Cosine</SelectItem>
                <SelectItem value="polynomial">Polynomial</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Advanced Options</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="mixed-precision">Mixed Precision Training</Label>
            <Switch id="mixed-precision" />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="gradient-checkpointing">Gradient Checkpointing</Label>
            <Switch id="gradient-checkpointing" />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="ema">Use EMA</Label>
            <Switch id="ema" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
