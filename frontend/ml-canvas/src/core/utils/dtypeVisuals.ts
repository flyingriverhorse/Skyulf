import { Hash, Tag, ToggleLeft, Calendar, AlignLeft, Type as TypeIcon, type LucideIcon } from 'lucide-react';
import type { ColumnDtype } from '../types/edaProfile';

/**
 * Single source of truth for "what color/icon represents this column
 * dtype" across the EDA surface. Previously `VariableCard.tsx`,
 * `VariableRow.tsx`, and `JobsHistoryModal.tsx` each hand-rolled their
 * own dtype -> color switch, and had drifted into three different,
 * inconsistent palettes (e.g. DateTime was green in one file, cyan in
 * another; Categorical was purple in one, orange in another). This
 * picks one deliberate palette and every consumer shares it.
 *
 * Palette rationale: five visually-distinct hues spread across the
 * wheel, avoiding colors this app already uses for status semantics
 * elsewhere (red = destructive/error, green = success/"healthy").
 *   - Numeric     -> blue    (the conventional "numbers" color; already
 *                              the dominant choice before this change)
 *   - Categorical -> purple  (distinct "grouping" color)
 *   - Boolean     -> amber   (reads as a flag/switch, distinct from
 *                              the other four)
 *   - DateTime    -> teal    (fresh, clearly separate from blue/green)
 *   - Text        -> slate   (neutral — free-form text isn't really a
 *                              "structured" category like the other four)
 */

// `edaProfile.ts` types `ColumnProfile.dtype` as `ColumnDtype | string` (to
// tolerate backend values outside the known union), so callers pass either;
// accepting plain `string` here covers both without fighting eslint's
// `ban-types` rule over the `string & {}` autocomplete-widening idiom.
type DtypeKey = string;

interface DtypeVisual {
  /** Full badge className (bg + text, light + dark) for pill/chip badges. */
  badgeClass: string;
  /** Icon-only color class, for a bare icon with no background. */
  iconColorClass: string;
  /** Icon component to render for this dtype. */
  Icon: LucideIcon;
}

const DTYPE_VISUALS: Record<ColumnDtype, DtypeVisual> = {
  Numeric: {
    badgeClass: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
    iconColorClass: 'text-blue-500',
    Icon: Hash,
  },
  Categorical: {
    badgeClass: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
    iconColorClass: 'text-purple-500',
    Icon: Tag,
  },
  Boolean: {
    badgeClass: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
    iconColorClass: 'text-amber-500',
    Icon: ToggleLeft,
  },
  DateTime: {
    badgeClass: 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300',
    iconColorClass: 'text-teal-500',
    Icon: Calendar,
  },
  Text: {
    badgeClass: 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300',
    iconColorClass: 'text-slate-500',
    Icon: AlignLeft,
  },
};

// Hex equivalents of the same palette (Tailwind's `-500` shade) for chart
// libraries (Recharts/Plotly) that need a literal color value rather than
// a className.
const DTYPE_HEX_COLORS: Record<ColumnDtype, string> = {
  Numeric: '#3b82f6', // blue-500
  Categorical: '#a855f7', // purple-500
  Boolean: '#f59e0b', // amber-500
  DateTime: '#14b8a6', // teal-500
  Text: '#64748b', // slate-500
};

const FALLBACK_HEX_COLOR = '#6b7280'; // gray-500

const FALLBACK_VISUAL: DtypeVisual = {
  badgeClass: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300',
  iconColorClass: 'text-gray-500',
  Icon: TypeIcon,
};

function resolve(dtype: DtypeKey): DtypeVisual {
  return DTYPE_VISUALS[dtype as ColumnDtype] ?? FALLBACK_VISUAL;
}

/** Full pill/badge className (background + text, light + dark) for a dtype. */
export function getDtypeBadgeClass(dtype: DtypeKey): string {
  return resolve(dtype).badgeClass;
}

/** Bare icon color class (no background) for a dtype. */
export function getDtypeIconColorClass(dtype: DtypeKey): string {
  return resolve(dtype).iconColorClass;
}

/** Icon component for a dtype (render as `<DtypeIcon dtype={...} className="..." />`). */
export function getDtypeIcon(dtype: DtypeKey): LucideIcon {
  return resolve(dtype).Icon;
}

/** Literal hex color for a dtype, for chart libraries (Recharts/Plotly) that need a real color value. */
export function getDtypeHexColor(dtype: DtypeKey): string {
  return DTYPE_HEX_COLORS[dtype as ColumnDtype] ?? FALLBACK_HEX_COLOR;
}
