import { ChevronDown, ChevronRight } from 'lucide-react';

interface Props {
  label: string;
  columnCount: number;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
}

export function SectionHeader({ label, columnCount, expanded, onExpandedChange }: Props) {
  return (
    <tr
      className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
      onClick={() => { onExpandedChange(!expanded); }}
    >
      <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={columnCount}>
        {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {label}
      </td>
    </tr>
  );
}
