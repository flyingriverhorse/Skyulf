# Why Data Preview Node Triggers Background Execution

## Special Case: data_preview (Data Snapshot Node)

The **Data Preview** node is unique - it's treated as a **transformation-capable node** even though it's primarily for viewing data.

## The Reason

### User Workflow with Data Preview:
```
1. User opens Data Preview node
2. Sees sample data (1000 rows) - FAST
3. Clicks "Save Changes"
   ↓
4. Background: Full dataset loads
   ↓
5. User clicks "Refresh Full Dataset" button
   ↓
6. Full dataset ALREADY READY - INSTANT display! ⚡
```

### Without Background Pre-loading:
```
1. User opens Data Preview node
2. Sees sample data (1000 rows)
3. Clicks "Save Changes"
4. Later, clicks "Refresh Full Dataset" button
   ↓
5. Wait 15-30 seconds for full dataset to load ❌
   ↓
6. User frustrated by delay
```

## Classification

### ✅ Data Preview (data_preview)
- **Primary purpose**: View dataset
- **Special capability**: Can refresh to full dataset
- **Triggers background**: YES ✅
- **Reason**: Pre-loads full dataset for instant "Refresh Full Dataset"

### ❌ Dataset Profile (dataset_profile)
- **Primary purpose**: View statistics
- **Special capability**: None (always uses samples)
- **Triggers background**: NO ❌
- **Reason**: Lightweight profiling, samples sufficient

### ❌ Binned Distribution (binned_distribution)
- **Primary purpose**: Visualize bins
- **Special capability**: None (visualization only)
- **Triggers background**: NO ❌
- **Reason**: Visualization, samples sufficient

### ❌ Skewness Distribution (skewness_distribution)
- **Primary purpose**: Visualize skewness
- **Special capability**: None (visualization only)
- **Triggers background**: NO ❌
- **Reason**: Visualization, samples sufficient

## Code Implementation

### Before (Incorrect):
```typescript
// data_preview was in INSPECTION_NODE_TYPES
const INSPECTION_NODE_TYPES = new Set([
  'binned_distribution',
  'data_preview',        // ❌ WRONG - prevents background execution
  'outlier_monitor',
  'skewness_distribution',
  'dataset_profile',
]);
```

### After (Correct):
```typescript
// data_preview removed from INSPECTION_NODE_TYPES
const INSPECTION_NODE_TYPES = new Set([
  'binned_distribution',
  // 'data_preview' removed - needs full dataset capability
  'outlier_monitor',
  'skewness_distribution',
  'dataset_profile',
]);
```

## User Experience Impact

### With Background Pre-loading (Current):
```
Timeline:
--------
00:00 - Open Data Preview node
00:01 - See sample (1000 rows) - FAST
00:02 - Click "Save Changes"
        └─→ Background: Full dataset loading starts
00:03 - Continue working on other nodes
00:10 - Come back to Data Preview
00:11 - Click "Refresh Full Dataset"
        └─→ INSTANT! Data already loaded ⚡

Total wait time: 0 seconds when refreshing
```

### Without Background Pre-loading (Old):
```
Timeline:
--------
00:00 - Open Data Preview node
00:01 - See sample (1000 rows)
00:02 - Click "Save Changes"
        └─→ No background loading
00:03 - Continue working
00:10 - Come back to Data Preview
00:11 - Click "Refresh Full Dataset"
        └─→ Wait for full dataset load...
00:26 - Finally see full dataset (15 second delay) ❌

Total wait time: 15 seconds of staring at loading spinner
```

## Summary

| Node | Type | Background Trigger? | Why? |
|------|------|---------------------|------|
| data_preview | Dataset Viewer | ✅ YES | Has "Refresh Full Dataset" button - pre-load for instant access |
| dataset_profile | Statistics | ❌ NO | Lightweight profiling, always uses samples |
| binned_distribution | Visualization | ❌ NO | Chart only, samples sufficient |
| skewness_distribution | Visualization | ❌ NO | Chart only, samples sufficient |

## Design Decision

**data_preview is a hybrid node:**
- Acts like inspection node for opening (fast samples)
- Acts like transformation node for saving (background full dataset)
- Best of both worlds: Fast initial view + instant full dataset refresh

This gives users:
1. ⚡ **Fast opening** - samples load in 1 second
2. 🚀 **Instant full dataset** - when clicking refresh button
3. 💪 **No waiting** - background pre-loading while working
4. ✅ **Smooth UX** - never blocked by loading

---

**Last Updated:** October 20, 2025  
**Status:** ✅ Implemented  
**Impact:** Data Preview "Refresh Full Dataset" now instant!
