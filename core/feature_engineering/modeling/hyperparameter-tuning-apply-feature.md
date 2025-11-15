# Hyperparameter Tuning Apply Feature

## Overview
This feature enables the Train Model node to automatically detect and apply the best hyperparameters from successful tuning runs for the currently selected model type. Users no longer need to manually remember which model was tuned or switch contexts—the system intelligently checks for available tuned parameters for each model type and presents an "Apply Best Params" button when applicable.

## Problem Solved
Previously, even after running hyperparameter tuning and finding optimal parameters, users had to:
1. Remember which model type was tuned
2. Manually navigate to find the tuning results
3. Copy/paste parameters or switch between different node configurations
4. Had no indication when tuned parameters were available for a specific model

## Solution Implemented

### 1. Backend API Enhancement

#### New Endpoint: `/api/hyperparameter-tuning/best-params/{model_type}`

**Purpose**: Fetch the most recent successful hyperparameter tuning results for a specific model type.

**Parameters**:
- `model_type` (path): The model type to search for (e.g., "RandomForestClassifier", "LogisticRegression")
- `pipeline_id` (query, optional): Filter by specific pipeline
- `dataset_source_id` (query, optional): Filter by specific dataset

**Response**:
```json
{
  "available": true,
  "model_type": "RandomForestClassifier",
  "job_id": "abc123...",
  "pipeline_id": "pipeline_xyz",
  "node_id": "node_456",
  "run_number": 3,
  "best_params": {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5
  },
  "best_score": 0.9234,
  "scoring": "accuracy",
  "finished_at": "2025-10-27T10:30:00Z",
  "search_strategy": "random",
  "n_iterations": 20
}
```

**Implementation Details**:
- Queries the `hyperparameter_tuning_jobs` table through the new `core.feature_engineering.modeling.hyperparameter_tuning.jobs` package (service layer builds on `jobs/repository.py`).
- Filters by `model_type` and `SUCCEEDED` status
- Orders by most recent `finished_at` timestamp
- Returns the best parameters from the most recent successful run
- Implemented inside `core/feature_engineering/routes.py` under the hyperparameter-tuning API group

### 2. Frontend API Integration

#### New Function: `fetchBestHyperparameters()`

**Location**: `frontend/feature-canvas/src/api.ts`

**Purpose**: Client-side function to call the new backend endpoint

**Usage**:
```typescript
const bestParams = await fetchBestHyperparameters('RandomForestClassifier', {
  pipelineId: 'pipeline_xyz',
  datasetSourceId: 'dataset_123'
});
```

**Type Definition**:
```typescript
export type BestHyperparametersResponse = {
  available: boolean;
  model_type: string;
  message?: string;
  job_id?: string;
  pipeline_id?: string;
  node_id?: string;
  run_number?: number;
  best_params?: Record<string, any>;
  best_score?: number;
  scoring?: string;
  finished_at?: string;
  search_strategy?: string;
  n_iterations?: number;
};
```

### 3. UI Enhancements in Train Model Node

#### ModelTrainingSection Component Updates

**Location**: `frontend/feature-canvas/src/components/node-settings/nodes/modeling/ModelTrainingSection.tsx`

**Key Changes**:

1. **New Query Hook**: Added React Query hook to fetch best hyperparameters for the current model type:
   ```typescript
   const bestParamsQuery = useQuery<BestHyperparametersResponse, Error>({
     queryKey: ['best-hyperparameters', modelType, pipelineIdFromSavedConfig, sourceId],
     queryFn: () => fetchBestHyperparameters(modelType, {
       pipelineId: pipelineIdFromSavedConfig || undefined,
       datasetSourceId: sourceId || undefined,
     }),
     enabled: Boolean(modelType && sourceId),
     staleTime: 30 * 1000,
   });
   ```

2. **Smart Parameter Selection**: The system now prioritizes parameters from the new API endpoint, falling back to the legacy tuning job list for backward compatibility:
   ```typescript
   const bestParamsFromAPI = useMemo(() => {
     // Filter and validate params from the dedicated endpoint
   });
   
   const bestParamsToUse = bestParamsFromAPI || bestParamsFromLatestTuning;
   ```

3. **Model Type Matching**: Enhanced logic to check if tuned parameters match the currently selected model:
   ```typescript
   const tuningModelMatches = Boolean(
     modelType && (
       (bestParamsData?.available && bestParamsData?.model_type === modelType) ||
       (latestTuningJob && latestTuningJob.model_type === modelType)
     )
   );
   ```

4. **Visual Feedback**: The UI now shows a clear indicator when tuned parameters are available:
   ```
   ✓ Tuned parameters available for RandomForestClassifier (run 3) • 2 hours ago
   ```

5. **"Apply Best Params" Button**: 
   - Automatically appears when tuned parameters are available for the current model
   - Disabled when no matching parameters exist or model types don't match
   - Shows helpful warning messages when clicked but conditions aren't met
   - Styled with visual prominence to draw attention

## User Workflow

### Before This Feature:
1. User runs hyperparameter tuning on RandomForestClassifier
2. Tuning completes successfully with best parameters
3. User goes to Train Model node
4. User selects "RandomForestClassifier" from dropdown
5. **User has no indication that tuned parameters exist**
6. User must manually remember to check tuning history
7. User must find and copy parameters manually

### After This Feature:
1. User runs hyperparameter tuning on RandomForestClassifier
2. Tuning completes successfully with best parameters
3. User goes to Train Model node
4. User selects "RandomForestClassifier" from dropdown
5. **✓ System immediately shows: "Tuned parameters available for RandomForestClassifier (run 3)"**
6. **"Apply Best Params" button appears and is enabled**
7. User clicks button → parameters are automatically applied
8. Success message confirms: "Applied best parameters from tuning run 3"
9. Advanced mode is automatically opened to show the applied parameters

## Benefits

1. **Model-Specific Intelligence**: Checks for tuned parameters specific to each model type, not just the latest tuning job
2. **Seamless UX**: No need to switch between different model types to see tuning history
3. **Clear Visibility**: Users immediately see when tuned parameters are available
4. **One-Click Apply**: Single button click applies all optimal parameters
5. **Backward Compatible**: Falls back to legacy tuning job list if new API doesn't find results
6. **Context-Aware**: Filters by pipeline and dataset when available for more relevant results

## Technical Architecture

```
┌─────────────────────────────────────────┐
│   User Interface (Train Model Node)     │
│  - Model type dropdown                   │
│  - "Apply Best Params" button            │
│  - Visual indicators                     │
└──────────────┬──────────────────────────┘
               │
               │ React Query
               ├──────────────────────────┐
               │                          │
               ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐
│  New API Endpoint    │   │  Legacy Tuning Jobs  │
│  /best-params/       │   │  Endpoint            │
│  {model_type}        │   │  (fallback)          │
└──────────┬───────────┘   └──────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│    Database Query                        │
│  - Filter by model_type                  │
│  - Filter by SUCCESS status              │
│  - Order by finished_at DESC             │
│  - LIMIT 1                               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   hyperparameter_tuning_jobs Table      │
│  - id, model_type, status                │
│  - best_params, best_score               │
│  - finished_at, run_number               │
└─────────────────────────────────────────┘
```

## Example Scenarios

### Scenario 1: Tuned Parameters Available
- User selects "RandomForestClassifier"
- System finds recent tuning run with best_params
- Button appears: "Apply Best Params" (enabled)
- Message: "✓ Tuned parameters available for RandomForestClassifier (run 3) • 2 hours ago"

### Scenario 2: Different Model Type Tuned
- User selects "LogisticRegression"
- System finds tuning run but for "RandomForestClassifier"
- Button appears: "Apply Best Params" (disabled)
- Message: "Latest tuning run targeted RandomForestClassifier."

### Scenario 3: No Tuning History
- User selects "GradientBoostingClassifier"
- System finds no tuning history for this model
- No button appears
- User sees only "Default" / "Advanced" toggle

### Scenario 4: After Applying Parameters
- User clicks "Apply Best Params"
- Success message: "Applied best parameters from tuning run 3"
- Advanced mode automatically opens
- All tuned parameters are visible and editable
- Parameters saved with draft configuration

## Files Modified

1. **Backend**:
   - `core/feature_engineering/routes.py`: Added new endpoint and import

2. **Frontend**:
   - `frontend/feature-canvas/src/api.ts`: Added API function and types
   - `frontend/feature-canvas/src/components/node-settings/nodes/modeling/ModelTrainingSection.tsx`: Enhanced UI logic

## Future Enhancements

Potential improvements for future iterations:

1. **Parameter Comparison**: Show diff between current and tuned parameters
2. **Multiple Model History**: Show history of multiple tuning runs with selection
3. **Performance Metrics**: Display the score improvement from tuned parameters
4. **Auto-Apply Option**: Setting to automatically apply best parameters when available
5. **Parameter Origin Tags**: Visual tags on each parameter showing source (default/tuned/custom)
6. **Tuning Recommendations**: Suggest which parameters might benefit from further tuning

## Testing Recommendations

1. **Unit Tests**:
   - Test API endpoint with various model types
   - Test response when no tuning history exists
   - Test filtering by pipeline_id and dataset_source_id

2. **Integration Tests**:
   - Test end-to-end flow: tuning → apply → train
   - Test with multiple model types
   - Test backward compatibility with legacy tuning jobs

3. **UI Tests**:
   - Test button visibility conditions
   - Test parameter application flow
   - Test visual indicators and messages

4. **Edge Cases**:
   - Multiple successful tuning runs for same model
   - Tuning run with no best_params
   - Model type name changes
   - Concurrent tuning jobs

## Conclusion

This feature significantly improves the user experience by making hyperparameter tuning results immediately accessible and applicable in the Train Model node. Users can now seamlessly move from tuning to training with optimal parameters, reducing friction and potential errors in the MLOps workflow.
