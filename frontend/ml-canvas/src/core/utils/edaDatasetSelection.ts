/**
 * Dataset-selection resolution for the EDA page.
 *
 * Extracted as pure functions so the precedence rules are testable without
 * mounting the page: an explicit `?dataset_id=` always beats whatever the
 * (module-singleton) EDA store happens to be holding, and the first-dataset
 * fallback only applies when nothing at all is selected.
 */

/** Minimal shape needed to pick a fallback dataset; the API returns richer objects. */
export interface DatasetOptionLike {
  id: number | string;
}

/** Dataset ids are positive integers; anything else in the URL is user-editable noise. */
function parseDatasetId(raw: string | null | undefined): number | null {
  if (raw == null) return null;
  const trimmed = raw.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const parsed = Number(trimmed);
  return parsed > 0 ? parsed : null;
}

/**
 * Decide which dataset the EDA page should be showing.
 *
 * @param paramValue - Raw `dataset_id` search-param value, if any.
 * @param current - Dataset currently held by the EDA store.
 * @param datasets - Loaded dataset list, used only for the initial fallback.
 * @returns The dataset id to select, or `null` when there is nothing to select.
 */
export function resolveEdaDatasetSelection(
  paramValue: string | null | undefined,
  current: number | null,
  datasets: readonly DatasetOptionLike[],
): number | null {
  const requested = parseDatasetId(paramValue);
  if (requested !== null) return requested;
  if (current !== null) return current;
  const first = datasets[0];
  return first === undefined ? null : parseDatasetId(String(first.id));
}

/**
 * Whether the resolved selection points at a dataset the EDA page cannot offer.
 *
 * A controlled `<select>` whose value matches no option falls back to the first
 * enabled option in the DOM, so without this check the page would display one
 * dataset's name while querying a completely different id.
 *
 * @param selected - Currently resolved dataset id.
 * @param datasets - Datasets returned by the usable-datasets query.
 * @param datasetsLoaded - `false` while the query is still in flight, so an
 *   empty list isn't mistaken for "this dataset doesn't exist".
 */
export function isSelectionMissingFromDatasets(
  selected: number | null,
  datasets: readonly DatasetOptionLike[],
  datasetsLoaded: boolean,
): boolean {
  if (selected === null || !datasetsLoaded) return false;
  return !datasets.some((ds) => Number(ds.id) === selected);
}

/**
 * Whether the URL should be rewritten to advertise the current selection, so
 * that reloads, shares, and back/forward all resolve to the same dataset.
 *
 * A usable `dataset_id` is always left alone: it is the authoritative input to
 * {@link resolveEdaDatasetSelection}, and overwriting it from a selection that
 * has not reconciled yet would discard the incoming deep link.
 */
export function shouldSyncDatasetParam(
  paramValue: string | null | undefined,
  selected: number | null,
): boolean {
  if (selected === null) return false;
  return parseDatasetId(paramValue) === null;
}
