import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Bug } from 'lucide-react';

interface DebugConfig {
  message: string;
}

const DebugSettings: React.FC<{ config: DebugConfig; onChange: (c: DebugConfig) => void }> = ({
  config,
  onChange,
}) => {
  return (
    <div className="p-4 space-y-2">
      <label className="block text-sm font-medium">Debug Message</label>
      <input
        type="text"
        className="w-full p-2 border rounded"
        value={config.message}
        onChange={(e) => onChange({ ...config, message: e.target.value })}
      />
    </div>
  );
};

export const DebugNode: NodeDefinition<DebugConfig> = {
  type: 'debug_node',
  label: 'Debug Node',
  category: 'Utility',
  description: 'A simple node for testing the registry.',
  icon: Bug,
  inputs: [{ id: 'in', label: 'Input', type: 'any' }],
  outputs: [{ id: 'out', label: 'Output', type: 'any' }],
  component: ({ data }) => (
    <div className="text-xs">
      Msg: <span className="font-mono">{data.message || '...'}</span>
    </div>
  ),
  settings: DebugSettings,
  validate: (config) => {
    return {
      isValid: config.message.length > 0,
      message: config.message.length === 0 ? 'Message is required' : undefined,
    };
  },
  getDefaultConfig: () => ({
    message: 'Hello World',
  }),
};
