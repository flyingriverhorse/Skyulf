import { useCallback, useEffect, useState } from 'react';
import {
	fetchOutlierRecommendations,
	type FeatureGraph,
	type OutlierRecommendationsResponse,
} from '../../../../api';
import type { CatalogFlagMap } from '../core/useCatalogFlags';

type GraphContext = FeatureGraph | null;

type UseOutlierRecommendationsArgs = {
	catalogFlags: CatalogFlagMap;
	sourceId?: string | null;
	hasReachableSource: boolean;
	graphContext: GraphContext;
	targetNodeId: string | null;
};

type UseOutlierRecommendationsResult = {
	outlierData: OutlierRecommendationsResponse | null;
	outlierError: string | null;
	isFetchingOutliers: boolean;
	refreshOutliers: () => void;
};

export const useOutlierRecommendations = ({
	catalogFlags,
	sourceId,
	hasReachableSource,
	graphContext,
	targetNodeId,
}: UseOutlierRecommendationsArgs): UseOutlierRecommendationsResult => {
	const { isOutlierNode } = catalogFlags;
	const [outlierData, setOutlierData] = useState<OutlierRecommendationsResponse | null>(null);
	const [outlierError, setOutlierError] = useState<string | null>(null);
	const [isFetchingOutliers, setIsFetchingOutliers] = useState(false);
	const [requestId, setRequestId] = useState(0);

	const refreshOutliers = useCallback(() => {
		setRequestId((previous) => previous + 1);
	}, []);

	useEffect(() => {
		let isActive = true;

		if (!isOutlierNode) {
			setOutlierData(null);
			setOutlierError(null);
			setIsFetchingOutliers(false);
			return () => {
				isActive = false;
			};
		}

		if (!sourceId) {
			setOutlierData(null);
			setOutlierError('Select a dataset to load outlier insights.');
			setIsFetchingOutliers(false);
			return () => {
				isActive = false;
			};
		}

		if (!hasReachableSource) {
			setOutlierData(null);
			setOutlierError('Connect this step to an upstream output to load outlier insights.');
			setIsFetchingOutliers(false);
			return () => {
				isActive = false;
			};
		}

		setIsFetchingOutliers(true);
		setOutlierError(null);

		fetchOutlierRecommendations(sourceId, {
			graph: graphContext,
			targetNodeId,
		})
			.then((result: OutlierRecommendationsResponse | null | undefined) => {
				if (!isActive) {
					return;
				}
				setOutlierData(result ?? null);
			})
			.catch((error: any) => {
				if (!isActive) {
					return;
				}
				setOutlierData(null);
				setOutlierError(error?.message ?? 'Unable to load outlier insights');
			})
			.finally(() => {
				if (isActive) {
					setIsFetchingOutliers(false);
				}
			});

		return () => {
			isActive = false;
		};
	}, [graphContext, hasReachableSource, isOutlierNode, requestId, sourceId, targetNodeId]);

	return {
		outlierData,
		outlierError,
		isFetchingOutliers,
		refreshOutliers,
	};
};
