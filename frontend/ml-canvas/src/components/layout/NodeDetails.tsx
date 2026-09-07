import { useState } from 'react';
import { Copy, Check, Info } from 'lucide-react';
import * as Popover from '@radix-ui/react-popover';

/** Keep troubleshooting identifiers available without expanding the settings header. */
export function NodeDetails({ nodeId }: { nodeId: string }) {
  const [copyStatus, setCopyStatus] = useState('');
  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(nodeId);
      setCopyStatus('Node ID copied.');
    } catch {
      setCopyStatus('Could not copy. Select the ID to copy it manually.');
    }
  };

  return <Popover.Root>
    <Popover.Trigger asChild>
      <button type="button" aria-label="Node information" title="Node information"
        className="shrink-0 rounded-md p-1.5 bg-primary/10 text-primary hover:bg-primary/20 focus-ring">
        <Info size={16} />
      </button>
    </Popover.Trigger>
    <Popover.Portal>
    <Popover.Content aria-label="Node information" side="bottom" align="start" sideOffset={8} collisionPadding={12}
      className="nokey z-50 w-72 max-w-[calc(100vw-24px)] space-y-3 rounded-lg border bg-popover p-3 text-xs text-popover-foreground shadow-lg">
      <p className="font-semibold">Node information</p>
      <dl className="space-y-2">
        <div>
          <dt className="mb-1 font-medium text-muted-foreground">Node ID</dt>
          <dd className="flex items-start gap-2">
            <code className="min-w-0 flex-1 break-all select-text">{nodeId}</code>
            <button type="button" onClick={() => { void copyId(); }} aria-label="Copy node ID" title="Copy node ID"
              className="shrink-0 rounded p-1.5 action-secondary focus-ring">
              {copyStatus === 'Node ID copied.' ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </dd>
        </div>
      </dl>
      <p className="text-muted-foreground">Use this ID when reporting a problem.</p>
      <p role="status" className="text-muted-foreground empty:hidden">{copyStatus}</p>
    </Popover.Content>
    </Popover.Portal>
  </Popover.Root>;
}
