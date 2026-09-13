/** Compare plain configuration values without conflating types or object key order. */
export function equalConfigValue(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (!isConfigContainer(a) || !isConfigContainer(b)) return false;
  if (Object.getPrototypeOf(a) !== Object.getPrototypeOf(b)) return false;
  if (Array.isArray(a) && a.length !== b.length) return false;
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) return false;
  return keys.every(key => Object.hasOwn(b, key) && equalConfigValue(a[key], b[key]));
}

/** Non-configuration objects remain changes unless the caller reuses their identity. */
function isConfigContainer(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object') return false;
  const prototype = Object.getPrototypeOf(value);
  return Array.isArray(value) || prototype === Object.prototype || prototype === null;
}
