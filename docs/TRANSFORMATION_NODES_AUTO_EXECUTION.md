# Dataset Sampling Strategy for Node Operations

## Overview
The system uses **smart sampling** to balance speed and accuracy:
- **Opening nodes**: Fast sample-based previews (1000 rows)
- **Full dataset execution**: Only when explicitly needed or automatically triggered for large datasets

## When Does Full Dataset Load?

### ✅ Automatic Full Dataset Execution
Full dataset loads automatically in these scenarios:

1. **Data Preview Node with "Refresh Full Dataset" button clicked**
   - User explicitly requests full dataset view
   - Button available in Data Preview node settings

2. **Background Jobs for Large Datasets**
   - When dataset > 200k rows
   - Runs asynchronously without blocking UI
   - Status polling shows progress

### ❌ Sample Data Only (Fast Operations)
Sample data (1000 rows) used for:

1. **Opening ANY node settings**
   - Fast recommendations (encoding, scaling, etc.)
   - Quick column statistics
   - Responsive UI

2. **Inspection nodes**
   - `data_preview` - Dataset snapshot viewer
   - `binned_distribution` - Binned visualizations
   - `skewness_distribution` - Skewness visualizations  
   - `dataset_profile` - Lightweight profiling

3. **Configuration and recommendations**
   - All recommendation endpoints use samples
   - Fast parameter suggestions
   - Quick column analysis

## How It Works

### Opening a Node (FAST - Uses Samples)
```
User: Opens "Label Encoding" node
↓
Frontend: Calls preview_pipeline with default sample_size=1000
↓
Backend: Loads 1000 sample rows
↓
Backend: Generates encoding recommendations from sample
↓
Frontend: Shows recommendations in ~1-2 seconds
↓
User: Configures encoding, clicks Save
↓
Node saved with configuration (no full dataset load)
Node saved with configuration (no full dataset load)
```

### Full Dataset Execution (When Explicitly Requested)
```
User: Opens "Data Preview" node
↓
Frontend: Shows sample preview (1000 rows)
↓
User: Clicks "Refresh Full Dataset" button
↓
Frontend: Calls preview_pipeline with sample_size=0
↓
Backend: Checks dataset size
↓
If < 200k rows:
  → Loads full dataset immediately
  → Returns complete preview
If > 200k rows:
  → Creates background job
  → Returns job ID
  → Frontend polls for status
  → Shows progress updates
```

### Pipeline Execution at Runtime
```
User: Completes pipeline configuration
↓
User: Triggers training/export
↓
Backend: Loads FULL dataset (sample_size=0)
↓
Backend: Applies ALL transformations to full data
↓
Backend: Trains model OR exports transformed data
↓
Result: ✅ Full dataset properly transformed
```

## Benefits of This Approach

### ✅ Fast UI/UX
- Node settings open instantly (1-2 seconds)
- Recommendations generate quickly
- No waiting for large datasets to load
- Responsive and smooth experience

### ✅ Accurate Recommendations  
- 1000 rows is statistically significant for most datasets
- Encoding suggestions representative of full data
- Scaling recommendations reliable from samples
- Binning strategies accurate with samples

### ✅ Data Integrity
- Full dataset transformations applied at execution time
- No sample-to-full inconsistencies
- Complete data used for training/export
- Production-ready results

## Implementation Details

### Code Logic
```python
# Default: Use sample for fast operations
requested_sample_size = int(payload.sample_size)  # Usually 1000
effective_sample_size = requested_sample_size

# Only load full dataset when explicitly requested
if requested_sample_size == 0:
    effective_sample_size = 0  # Full dataset
    # Trigger background job if > 200k rows
```

### Frontend Default
```typescript
// When opening nodes, use sample
const DEFAULT_PREVIEW_SAMPLE_SIZE = 1000;

// When refreshing full dataset in Data Preview node
previewRequest.sample_size = 0;  // Explicit full dataset request
```

## Common Scenarios

### Scenario 1: Configure Encoding (FAST)
```
1. Open Label Encoding node
2. System loads 1000 sample rows (~1 second)
3. Shows encoding recommendations
4. User configures columns
5. Clicks Save (instant)
```
**Time**: ~2-3 seconds total ✅

### Scenario 2: View Full Dataset (Background Job)
```
1. Open Data Preview node
2. See sample preview (1000 rows)
3. Click "Refresh Full Dataset"
4. If large: Background job starts
5. Poll status every 5 seconds
6. Full dataset ready in ~10-30 seconds
```
**Time**: Varies by dataset size, non-blocking ✅

### Scenario 3: Train Model (Full Dataset)
```
1. Configure pipeline with transformations
2. All nodes show sample-based previews (fast)
3. Click "Train Model"
4. Backend loads full dataset
5. Applies all transformations to full data
6. Trains on complete transformed dataset
```
**Time**: Happens at execution, complete data ✅

## Edge Cases

### Large Datasets (>200k rows)
- Opening nodes: Still fast (1000 sample)
- Full dataset request: Background job created
- Training: May take time, but uses full data
- Solution: Progress indicators, non-blocking UI

### Small Datasets (<1000 rows)
- Sample size = actual dataset size
- No performance penalty
- Full dataset loaded even for previews

### Very Large Datasets (>1M rows)
- Recommendations from 1000-row sample (fast)
- Full dataset execution via background jobs
- Memory-efficient chunked processing
- Status polling for user feedback

## Configuration

### File Modified
- `core/feature_engineering/routes.py` (preview_pipeline endpoint)

### Key Variables
- `requested_sample_size`: From frontend (default 1000)
- `effective_sample_size`: Actual rows to load
- `FULL_DATASET_EXECUTION_ROW_LIMIT`: 200,000 rows (background job threshold)
- `DEFAULT_SAMPLE_CAP`: Safety limit for data preview node

## Testing Scenarios

### Test 1: Fast Node Opening
```
Dataset (100k rows) → Open Label Encoding
Expected: Opens in 1-2 seconds with sample recommendations
```

### Test 2: Full Dataset Request
```
Dataset (50k rows) → Data Preview → Refresh Full Dataset
Expected: Loads full 50k rows, completes quickly
```

### Test 3: Large Dataset Background Job
```
Dataset (500k rows) → Data Preview → Refresh Full Dataset
Expected: Background job created, polling UI shows progress
```

### Test 4: Training Pipeline
```
Dataset (200k rows) → Cleaning → Encoding → Train Model
Expected: All nodes open fast, training uses full 200k rows
```

## Migration Notes

### No Breaking Changes
- Existing behavior preserved
- Opening nodes remains fast (samples)
- Full dataset available when needed
- Background jobs for large datasets

### Performance Impact
- ✅ Faster node opening (samples only)
- ✅ Quick recommendations
- ✅ Responsive UI
- ✅ Full dataset at execution time

## Related Documentation
- `BACKGROUND_JOB_POLLING.md` - How polling works for large datasets
- Frontend: `DataSnapshotSection.tsx` - Full execution UI with polling
- Frontend: `usePipelinePreview.ts` - Preview hook with sample defaults

---

**Last Updated:** October 20, 2025  
**Status:** ✅ Optimized for Speed and Accuracy  
**Key Principle:** Samples for configuration, full dataset for execution
