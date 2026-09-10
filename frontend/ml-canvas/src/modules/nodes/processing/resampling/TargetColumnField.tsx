import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import * as Popover from '@radix-ui/react-popover';
import { ValidationField } from '../../../../components/shared/ValidationField';

/** Keep editable target suggestions anchored to their input across panel resizing and scrolling. */
export function TargetColumnField({ id, value, columns, onChange }: {
  id: string; value: string; columns: string[]; onChange: (value: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const matches = columns.filter(column => column.toLowerCase().includes(value.toLowerCase()));
  const isOpen = open && matches.length > 0;
  const listId = `${id}-target-column-suggestions`;
  const activeColumn = matches[activeIndex];
  const activeId = `${listId}-${activeIndex}`;

  useEffect(() => {
    if (isOpen) document.getElementById(activeId)?.scrollIntoView?.({ block: 'nearest' });
  }, [activeId, isOpen]);

  const select = (column: string) => {
    onChange(column);
    setOpen(false);
    setActiveIndex(-1);
  };

  const navigate = (event: KeyboardEvent<HTMLInputElement>, direction: number) => {
    event.preventDefault();
    if (!matches.length) return;
    setOpen(true);
    setActiveIndex(index => index < 0
      ? (direction > 0 ? 0 : matches.length - 1)
      : (index + direction + matches.length) % matches.length);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    switch (event.key) {
      case 'ArrowDown': navigate(event, 1); break;
      case 'ArrowUp': navigate(event, -1); break;
      case 'Enter':
        if (isOpen && activeColumn !== undefined) {
          event.preventDefault();
          select(activeColumn);
        }
        break;
      case 'Escape':
        if (isOpen) {
          event.preventDefault();
          event.stopPropagation();
          setOpen(false);
        }
        break;
    }
  };

  return <Popover.Root open={isOpen} onOpenChange={setOpen}>
    <ValidationField field="target_column">
      <Popover.Anchor asChild>
        <input ref={inputRef} aria-label="Target Column" role="combobox" type="text"
          aria-autocomplete="list" aria-expanded={isOpen} aria-controls={isOpen ? listId : undefined}
          aria-activedescendant={isOpen && activeColumn !== undefined ? activeId : undefined}
          autoComplete="off" placeholder="e.g., target" value={value}
          className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          onFocus={() => setOpen(true)} onClick={() => setOpen(true)} onKeyDown={handleKeyDown}
          onChange={event => { onChange(event.target.value); setActiveIndex(-1); setOpen(true); }} />
      </Popover.Anchor>
    </ValidationField>
    <Popover.Portal>
      <Popover.Content id={listId} role="listbox" aria-label="Target column suggestions"
        side="bottom" align="start" sideOffset={4} collisionPadding={8}
        onOpenAutoFocus={event => event.preventDefault()}
        onCloseAutoFocus={event => event.preventDefault()}
        onInteractOutside={event => {
          if (event.detail.originalEvent.target === inputRef.current) event.preventDefault();
        }}
        className="z-50 overflow-y-auto rounded-md border bg-popover p-1 text-popover-foreground shadow-lg"
        style={{ width: 'var(--radix-popover-trigger-width)', maxHeight: 'min(192px, var(--radix-popover-content-available-height))' }}>
        {matches.map((column, index) => <button key={column} id={`${listId}-${index}`} type="button"
          role="option" tabIndex={-1} aria-selected={index === activeIndex}
          className="block w-full truncate rounded px-2 py-1.5 text-left text-sm hover:bg-accent aria-selected:bg-accent"
          title={column} onMouseDown={event => event.preventDefault()} onClick={() => select(column)}>
          {column}
        </button>)}
      </Popover.Content>
    </Popover.Portal>
  </Popover.Root>;
}
