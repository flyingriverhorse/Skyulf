import type { ModelHyperparameterField } from '../../../../api';

export type PrimaryMetric = {
	label: string;
	value: number;
};

export const STATUS_LABEL: Record<string, string> = {
	queued: 'Queued',
	running: 'Running',
	succeeded: 'Succeeded',
	failed: 'Failed',
	cancelled: 'Cancelled',
};

export const METRIC_PREFERENCE = {
	classification: ['accuracy', 'f1_weighted', 'roc_auc', 'precision_weighted', 'recall_weighted'],
	regression: ['rmse', 'mae', 'r2'],
	fallback: ['rmse', 'mae', 'r2', 'accuracy', 'f1_weighted', 'roc_auc'],
};

export const filterHyperparametersByFields = (
	values: Record<string, any> | null | undefined,
	fieldNames: Set<string>
): Record<string, any> => {
	if (!values) {
		return {};
	}
	const filtered: Record<string, any> = {};
	Object.entries(values).forEach(([key, value]) => {
		if (fieldNames.has(key)) {
			filtered[key] = value;
		}
	});
	return filtered;
};

export const valuesEqual = (first: any, second: any): boolean => {
	if (first === second) {
		return true;
	}
	if (typeof first === 'number' && typeof second === 'number' && Number.isNaN(first) && Number.isNaN(second)) {
		return true;
	}
	if (Array.isArray(first) && Array.isArray(second)) {
		if (first.length !== second.length) {
			return false;
		}
		return first.every((item, index) => valuesEqual(item, second[index]));
	}
	if (first && second && typeof first === 'object' && typeof second === 'object') {
		try {
			return JSON.stringify(first) === JSON.stringify(second);
		} catch (error) {
			return false;
		}
	}
	return false;
};

export const sanitizeHyperparametersForPayload = (
	values: Record<string, any> | null | undefined,
	fieldNames: Set<string>,
	fieldMap?: Record<string, ModelHyperparameterField>
): Record<string, any> => {
	if (!values) {
		return {};
	}
	const sanitized: Record<string, any> = {};
	Object.entries(values).forEach(([key, rawValue]) => {
		if (!fieldNames.has(key)) {
			return;
		}
		if (rawValue === '' || rawValue === null || rawValue === undefined) {
			return;
		}
		const field = fieldMap?.[key];
		let value = rawValue;
		if (field) {
			if (typeof value === 'string') {
				const trimmedValue = value.trim();
				if (!trimmedValue) {
					return;
				}
				if (field.type === 'select') {
					value = trimmedValue;
				}
			}

			if (field.name === 'multi_class' && typeof value === 'string') {
				const normalizedMulti = value.trim().toLowerCase();
				if (normalizedMulti === 'auto') {
					return;
				}
				if (normalizedMulti === 'ovr' || normalizedMulti === 'multinomial') {
					value = normalizedMulti;
				}
			}

			if (field.default !== undefined) {
				if (typeof field.default === 'string' && typeof value === 'string') {
					const defaultTrimmed = field.default.trim();
					const valueTrimmed = value.trim();
					if (defaultTrimmed === valueTrimmed) {
						return;
					}
				} else if (valuesEqual(field.default, value)) {
					return;
				}
			}
		}

		sanitized[key] = value;
	});
	return sanitized;
};

export const isRecord = (value: unknown): value is Record<string, any> => {
	return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
};

export const cloneJson = <T,>(value: T): T => {
	try {
		return JSON.parse(JSON.stringify(value));
	} catch (error) {
		return value;
	}
};

export const resolveMetricPreference = (metrics: Record<string, any> | null | undefined): string[] => {
	if (!metrics) {
		return METRIC_PREFERENCE.fallback;
	}
	const modelType = String(metrics.model_type ?? metrics.modelType ?? '').toLowerCase();
	if (modelType.includes('regressor')) {
		return METRIC_PREFERENCE.regression;
	}
	if (modelType.includes('classifier') || modelType.includes('classification')) {
		return METRIC_PREFERENCE.classification;
	}
	const hinted = String(metrics.problem_type ?? metrics.problemType ?? '').toLowerCase();
	if (hinted === 'regression') {
		return METRIC_PREFERENCE.regression;
	}
	if (hinted === 'classification') {
		return METRIC_PREFERENCE.classification;
	}
	return METRIC_PREFERENCE.fallback;
};

export const pickPrimaryMetric = (metrics: Record<string, any> | null | undefined): PrimaryMetric | null => {
	if (!metrics) {
		return null;
	}

	const metricBuckets: Array<{ dataset: Record<string, any>; labelPrefix: string }> = [];
	const cvMean = metrics?.cross_validation?.metrics?.mean;
	if (isRecord(cvMean) && metrics?.cross_validation?.status === 'completed') {
		metricBuckets.push({ dataset: cvMean, labelPrefix: 'CV mean ' });
	}

	const candidateBuckets = [metrics.test, metrics.validation, metrics.train];
	for (const bucket of candidateBuckets) {
		if (isRecord(bucket)) {
			metricBuckets.push({ dataset: bucket, labelPrefix: '' });
		}
	}

	const orderedKeys = resolveMetricPreference(metrics);

	for (const bucket of metricBuckets) {
		for (const key of orderedKeys) {
			const value = bucket.dataset[key];
			if (typeof value === 'number' && Number.isFinite(value)) {
				const label = bucket.labelPrefix ? `${bucket.labelPrefix}${key}` : key;
				return { label, value };
			}
		}
	}

	for (const bucket of metricBuckets) {
		for (const [key, value] of Object.entries(bucket.dataset)) {
			if (typeof value === 'number' && Number.isFinite(value)) {
				const label = bucket.labelPrefix ? `${bucket.labelPrefix}${key}` : key;
				return { label, value };
			}
		}
	}

	return null;
};

export const formatMetricValue = (value: number): string => {
	if (!Number.isFinite(value)) {
		return '';
	}
	if (Math.abs(value) >= 1000) {
		return value.toFixed(0);
	}
	if (Math.abs(value) >= 100) {
		return value.toFixed(1);
	}
	if (Math.abs(value) >= 10) {
		return value.toFixed(2);
	}
	return value.toFixed(3);
};
