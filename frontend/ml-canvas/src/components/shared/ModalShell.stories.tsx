import { useState } from 'react';
import { ModalShell } from './ModalShell';

export default { title: 'Shared / ModalShell' };

export const Basic = () => {
  const [open, setOpen] = useState(true);
  return (
    <>
      <button onClick={() => setOpen(true)} className="m-6 px-4 py-2 bg-blue-600 text-white rounded">
        Open modal
      </button>
      <ModalShell isOpen={open} onClose={() => setOpen(false)} title="Example modal">
        <div className="p-6 text-sm">
          Body content lives here. Press <kbd>Esc</kbd> or click outside to dismiss.
        </div>
      </ModalShell>
    </>
  );
};

export const WithFooter = () => {
  const [open, setOpen] = useState(true);
  return (
    <ModalShell
      isOpen={open}
      onClose={() => setOpen(false)}
      title="Confirm action"
      size="md"
      footer={
        <div className="flex justify-end gap-2 p-4 border-t border-slate-200 dark:border-slate-700">
          <button className="px-3 py-1.5 text-sm" onClick={() => setOpen(false)}>Cancel</button>
          <button className="px-3 py-1.5 text-sm bg-red-600 text-white rounded">Delete</button>
        </div>
      }
    >
      <div className="p-6 text-sm">Are you sure?</div>
    </ModalShell>
  );
};

export const LargeContent = () => {
  const [open, setOpen] = useState(true);
  return (
    <ModalShell isOpen={open} onClose={() => setOpen(false)} title="Long body" size="2xl">
      <div className="p-6 space-y-4 text-sm">
        {Array.from({ length: 30 }).map((_, i) => (
          <p key={i}>Paragraph {i + 1}: lorem ipsum dolor sit amet.</p>
        ))}
      </div>
    </ModalShell>
  );
};
