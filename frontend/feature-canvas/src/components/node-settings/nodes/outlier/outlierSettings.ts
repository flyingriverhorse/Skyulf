import type {
	OutlierColumnInsight,
	OutlierColumnStats,
	OutlierMethodDetail,
	OutlierMethodName,
	OutlierMethodSummary,
} from '../../../../api';

export type NormalizedOutlierConfig = {
	columns: string[];
	defaultMethod: OutlierMethodName;
	columnMethods: Record<string, OutlierMethodName>;
	autoDetect: boolean;
	skippedColumns: string[];
	methodParameters: Record<OutlierMethodName, Record<string, number>>;
	columnParameters: Record<string, Record<string, number>>;
};

export const OUTLIER_METHOD_ORDER: OutlierMethodName[] = ['iqr', 'zscore', 'elliptic_envelope', 'winsorize', 'manual'];

export const OUTLIER_METHOD_FALLBACK_LABELS: Record<OutlierMethodName, string> = {
	iqr: 'IQR filter',
	zscore: 'Z-score filter',
	elliptic_envelope: 'Elliptic Envelope',
	winsorize: 'Winsorize',
	manual: 'Manual bounds',
};

export const OUTLIER_PARAMETER_KEYS: Record<OutlierMethodName, string[]> = {
	iqr: ['multiplier'],
	zscore: ['threshold'],
	elliptic_envelope: ['contamination'],
	winsorize: ['lower_percentile', 'upper_percentile'],
	manual: ['lower_bound', 'upper_bound'],
};

export const OUTLIER_METHOD_DEFAULT_PARAMETERS: Record<OutlierMethodName, Record<string, number>> = {
	iqr: { multiplier: 1.5 },
	zscore: { threshold: 3 },
	elliptic_envelope: { contamination: 0.01 },
	winsorize: { lower_percentile: 5, upper_percentile: 95 },
	manual: {},
};

const clampNumber = (
	value: unknown,
	{ min, max }: { min?: number; max?: number } = {},
): number | null => {
	const numeric = typeof value === 'number' ? value : Number(value);
	if (!Number.isFinite(numeric)) {
		return null;
	}
	let result = numeric;
	if (typeof min === 'number' && Number.isFinite(min) && result < min) {
		result = min;
	}
	if (typeof max === 'number' && Number.isFinite(max) && result > max) {
		result = max;
	}
	return result;
};

const normalizeColumnList = (value: unknown): string[] => {
	const values: string[] = [];
	if (Array.isArray(value)) {
		value.forEach((entry) => {
			const column = String(entry ?? '').trim();
			if (column) {
				values.push(column);
			}
		});
	} else if (typeof value === 'string') {
		value
			.split(',')
			.map((entry) => entry.trim())
			.forEach((column) => {
				if (column) {
					values.push(column);
				}
			});
	}
	return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
};

const normalizeMethodParameters = (value: unknown): Record<OutlierMethodName, Record<string, number>> => {
	const result: Record<OutlierMethodName, Record<string, number>> = {
		iqr: { ...OUTLIER_METHOD_DEFAULT_PARAMETERS.iqr },
		zscore: { ...OUTLIER_METHOD_DEFAULT_PARAMETERS.zscore },
		elliptic_envelope: { ...OUTLIER_METHOD_DEFAULT_PARAMETERS.elliptic_envelope },
		winsorize: { ...OUTLIER_METHOD_DEFAULT_PARAMETERS.winsorize },
		manual: {},
	};

	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		return result;
	}

	Object.entries(value as Record<string, unknown>).forEach(([methodKey, parameters]) => {
		const normalizedMethod = methodKey.trim().toLowerCase() as OutlierMethodName;
		if (!OUTLIER_METHOD_ORDER.includes(normalizedMethod)) {
			return;
		}

		const current = result[normalizedMethod] ?? {};
		if (!parameters || typeof parameters !== 'object' || Array.isArray(parameters)) {
			return;
		}

		Object.entries(parameters as Record<string, unknown>).forEach(([parameterKey, rawValue]) => {
			const normalizedKey = parameterKey.trim().toLowerCase();
			if (!OUTLIER_PARAMETER_KEYS[normalizedMethod].includes(normalizedKey)) {
				return;
			}

			let bounds: { min?: number; max?: number } = {};
			if (normalizedKey === 'threshold') {
				bounds = { min: 0.1 };
			} else if (normalizedKey === 'multiplier') {
				bounds = { min: 0.1 };
			} else if (normalizedKey === 'lower_percentile' || normalizedKey === 'upper_percentile') {
				bounds = { min: 0, max: 100 };
			}

			const numeric = clampNumber(rawValue, bounds);
			if (numeric === null) {
				return;
			}
			current[normalizedKey] = numeric;
		});

		if (normalizedMethod === 'winsorize') {
			const lower = clampNumber(current.lower_percentile, { min: 0, max: 100 });
			const upper = clampNumber(current.upper_percentile, { min: 0, max: 100 });
			if (typeof lower === 'number' && typeof upper === 'number' && lower >= upper) {
				current.lower_percentile = OUTLIER_METHOD_DEFAULT_PARAMETERS.winsorize.lower_percentile;
				current.upper_percentile = OUTLIER_METHOD_DEFAULT_PARAMETERS.winsorize.upper_percentile;
			} else {
				if (typeof lower === 'number') {
					current.lower_percentile = lower;
				}
				if (typeof upper === 'number') {
					current.upper_percentile = upper;
				}
			}
		}

		result[normalizedMethod] = current;
	});

	return result;
};

const normalizeColumnParameters = (value: unknown): Record<string, Record<string, number>> => {
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		return {};
	}

	const result: Record<string, Record<string, number>> = {};

	Object.entries(value as Record<string, unknown>).forEach(([columnKey, parameters]) => {
		const column = String(columnKey ?? '').trim();
		if (!column || !parameters || typeof parameters !== 'object' || Array.isArray(parameters)) {
			return;
		}

		const normalizedParameters: Record<string, number> = {};

		Object.entries(parameters as Record<string, unknown>).forEach(([parameterKey, rawValue]) => {
			const normalizedKey = parameterKey.trim().toLowerCase();
			const permitted = Object.values(OUTLIER_PARAMETER_KEYS).some((keys) => keys.includes(normalizedKey));
			if (!permitted) {
				return;
			}

			let bounds: { min?: number; max?: number } = {};
			if (normalizedKey === 'threshold') {
				bounds = { min: 0.1 };
			} else if (normalizedKey === 'multiplier') {
				bounds = { min: 0.1 };
			} else if (normalizedKey === 'contamination') {
				bounds = { min: 0.0001, max: 0.49 };
			} else if (normalizedKey === 'lower_percentile' || normalizedKey === 'upper_percentile') {
				bounds = { min: 0, max: 100 };
			}

			const numeric = clampNumber(rawValue, bounds);
			if (numeric === null) {
				return;
			}
			normalizedParameters[normalizedKey] = numeric;
		});

		if (Object.keys(normalizedParameters).length) {
			result[column] = normalizedParameters;
		}
	});

	return result;
};

export const normalizeOutlierConfigValue = (value: unknown): NormalizedOutlierConfig => {
	const payload = value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

	const columns = normalizeColumnList(payload.columns);

	const rawDefault = typeof payload.default_method === 'string' ? payload.default_method.trim().toLowerCase() : '';
	const defaultMethod = OUTLIER_METHOD_ORDER.includes(rawDefault as OutlierMethodName)
		? (rawDefault as OutlierMethodName)
		: 'iqr';

	const columnMethods: Record<string, OutlierMethodName> = {};
	if (payload.column_methods && typeof payload.column_methods === 'object' && !Array.isArray(payload.column_methods)) {
		Object.entries(payload.column_methods as Record<string, unknown>).forEach(([columnKey, methodValue]) => {
			const column = String(columnKey ?? '').trim();
			const method = typeof methodValue === 'string' ? methodValue.trim().toLowerCase() : '';
			if (column && OUTLIER_METHOD_ORDER.includes(method as OutlierMethodName)) {
				columnMethods[column] = method as OutlierMethodName;
			}
		});
	}

	const autoDetect = typeof payload.auto_detect === 'boolean' ? payload.auto_detect : true;
	const skippedColumns = normalizeColumnList(payload.skipped_columns);
	const methodParameters = normalizeMethodParameters(payload.method_parameters);
	const columnParameters = normalizeColumnParameters(payload.column_parameters);

	return {
		columns,
		defaultMethod,
		columnMethods,
		autoDetect,
		skippedColumns,
		methodParameters,
		columnParameters,
	};
};

export type OutlierMethodOption = {
	value: OutlierMethodName;
	label: string;
};

export const createOutlierMethodLabelMap = (methods?: OutlierMethodDetail[] | null): Record<OutlierMethodName, string> => {
	const result: Record<OutlierMethodName, string> = { ...OUTLIER_METHOD_FALLBACK_LABELS };
	if (Array.isArray(methods)) {
		methods.forEach((detail) => {
			const key = typeof detail?.key === 'string' ? (detail.key.trim().toLowerCase() as OutlierMethodName) : null;
			const label = typeof detail?.label === 'string' ? detail.label.trim() : '';
			if (key && label && OUTLIER_METHOD_ORDER.includes(key)) {
				result[key] = label;
			}
		});
	}
	return result;
};

export const createOutlierMethodDetailMap = (methods?: OutlierMethodDetail[] | null): Map<OutlierMethodName, OutlierMethodDetail> => {
	const map = new Map<OutlierMethodName, OutlierMethodDetail>();
	if (Array.isArray(methods)) {
		methods.forEach((detail) => {
			const key = typeof detail?.key === 'string' ? (detail.key.trim().toLowerCase() as OutlierMethodName) : null;
			if (key && OUTLIER_METHOD_ORDER.includes(key)) {
				map.set(key, detail);
			}
		});
	}
	return map;
};

export const createOutlierMethodOptions = (labelMap: Record<OutlierMethodName, string>): OutlierMethodOption[] =>
	OUTLIER_METHOD_ORDER.map((method) => ({
		value: method,
		label: labelMap[method] ?? OUTLIER_METHOD_FALLBACK_LABELS[method] ?? method,
	}));

export type OutlierMethodSummaryMap = Partial<Record<OutlierMethodName, OutlierMethodSummary>>;

export type OutlierRecommendationRow = {
	column: string;
	dtype: string | null;
	stats: OutlierColumnStats;
	recommendedMethod: OutlierMethodName | null;
	recommendedLabel: string | null;
	recommendedReason: string | null;
	methodSummaries: OutlierMethodSummaryMap;
	currentOverride: OutlierMethodName | null;
	currentMethod: OutlierMethodName;
	currentMethodLabel: string;
	isOverrideApplied: boolean;
	isSelected: boolean;
	isSkipped: boolean;
	isExcluded: boolean;
	hasMissing: boolean;
};

type BuildOutlierRowsInput = {
	insights: OutlierColumnInsight[];
	outlierConfig: NormalizedOutlierConfig;
	labelMap: Record<OutlierMethodName, string>;
	selectedColumns: Set<string>;
	skippedColumns: Set<string>;
	excludedColumns: Set<string>;
};

export const buildOutlierRecommendationRows = ({
	insights,
	outlierConfig,
	labelMap,
	selectedColumns,
	skippedColumns,
	excludedColumns,
}: BuildOutlierRowsInput): OutlierRecommendationRow[] => {
	if (!insights.length) {
		return [];
	}

	const rows: OutlierRecommendationRow[] = [];

	insights.forEach((insight) => {
		const column = String(insight?.column ?? '').trim();
		if (!column) {
			return;
		}

		const methodSummaries: OutlierMethodSummaryMap = {};
		if (Array.isArray(insight?.method_summaries)) {
			insight.method_summaries.forEach((summary: OutlierMethodSummary) => {
				const method = summary?.method;
				if (method && OUTLIER_METHOD_ORDER.includes(method)) {
					methodSummaries[method] = summary;
				}
			});
		}

		const recommendedMethod = insight?.recommended_method && OUTLIER_METHOD_ORDER.includes(insight.recommended_method)
			? insight.recommended_method
			: null;

		const recommendedLabel = recommendedMethod ? labelMap[recommendedMethod] ?? OUTLIER_METHOD_FALLBACK_LABELS[recommendedMethod] ?? recommendedMethod : null;

		const columnOverride = outlierConfig.columnMethods[column] ?? null;
		const currentMethod = columnOverride ?? outlierConfig.defaultMethod;
		const currentMethodLabel = labelMap[currentMethod] ?? OUTLIER_METHOD_FALLBACK_LABELS[currentMethod] ?? currentMethod;
		const isSkipped = skippedColumns.has(column);
		const isExcluded = excludedColumns.has(column);
		const isSelected = selectedColumns.has(column);
		const hasMissing = Boolean(insight?.has_missing);

		rows.push({
			column,
			dtype: typeof insight?.dtype === 'string' ? insight.dtype : null,
			stats: insight?.stats ?? ({} as OutlierColumnStats),
			recommendedMethod,
			recommendedLabel,
			recommendedReason: typeof insight?.recommended_reason === 'string' ? insight.recommended_reason : null,
			methodSummaries,
			currentOverride: columnOverride,
			currentMethod,
			currentMethodLabel,
			isOverrideApplied: columnOverride !== null,
			isSelected,
			isSkipped,
			isExcluded,
			hasMissing,
		});
	});

	rows.sort((a, b) => a.column.localeCompare(b.column));
	return rows;
};

export const buildOutlierOverrideSummary = (
	overrideColumns: string[],
	columnMethods: Record<string, OutlierMethodName>,
	labelMap: Record<OutlierMethodName, string>,
	overrideCount: number,
): string | null => {
	if (!overrideColumns.length) {
		return null;
	}

	const preview = overrideColumns.map((column) => {
		const method = columnMethods[column];
		const label = method ? labelMap[method] ?? OUTLIER_METHOD_FALLBACK_LABELS[method] ?? method : null;
		return label ? `${column} -> ${label}` : column;
	});

	if (overrideCount > overrideColumns.length) {
		preview.push('…');
	}

	return preview.join(', ');
};
