/**
 * Parses a numeric `<input>` value, falling back to a default instead of
 * writing `NaN` into node config when the field is cleared or contains
 * a non-numeric value (e.g. mid-edit).
 */
export function parseIntSafe(value: string, fallback: number | undefined): number {
  const parsed = parseInt(value, 10);
  return Number.isNaN(parsed) ? (fallback ?? 0) : parsed;
}

/** Float variant of {@link parseIntSafe}. */
export function parseFloatSafe(value: string, fallback: number | undefined): number {
  const parsed = parseFloat(value);
  return Number.isNaN(parsed) ? (fallback ?? 0) : parsed;
}
