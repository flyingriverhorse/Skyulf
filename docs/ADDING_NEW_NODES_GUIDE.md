# Adding New Nodes - Background Execution Guide

## Quick Answer

**When adding new nodes, background execution is AUTOMATIC!** 

You only need to update configuration if the node is an **inspection node** (view-only, doesn't transform data).

## How It Works

### Automatic Background Execution

All nodes **automatically trigger background full dataset execution** by default when saved, EXCEPT nodes in the `INSPECTION_NODE_TYPES` set.

The logic in `NodeSettingsModal.tsx`:
```typescript
// Trigger full dataset execution in background after saving
// Skip for inspection nodes (they only view data, don't transform it)
if (sourceId && graphSnapshot && !isInspectionNode) {
  // Set loading status immediately
  if (onUpdateNodeData) {
    onUpdateNodeData(node.id, { backgroundExecutionStatus: 'loading' });
  }
  
  triggerFullDatasetExecution({...})
    .then(() => { /* success */ })
    .catch(() => { /* error */ });
}
```

### Inspection Nodes (Don't Trigger Background Execution)

These are defined in `catalogTypes.ts`:

```typescript
export const INSPECTION_NODE_TYPES = new Set<string>([
  'binned_distribution',
  'data_preview',
  'outlier_monitor',
  'skewness_distribution',
  'dataset_profile',
]);
```

## Adding a New Node

### Case 1: Transformation Node (Default - NOTHING TO CHANGE)

**If your node transforms data** (filtering, encoding, scaling, feature creation, etc.):

✅ **NO CODE CHANGES NEEDED!**

The node will automatically:
- Trigger background full dataset execution on save
- Show loading spinner while processing
- Show green checkmark when complete
- Show red error indicator if failed
- Clear status on reset

**Examples:**
- New encoding method: `custom_encoding`
- New feature: `text_vectorization`
- New transform: `outlier_capping`
- New filter: `advanced_filter`

### Case 2: Inspection Node (Add to INSPECTION_NODE_TYPES)

**If your node only VIEWS data** (doesn't transform it):

1. Open `frontend/feature-canvas/src/components/node-settings/utils/catalogTypes.ts`
2. Add your node's `catalogType` to `INSPECTION_NODE_TYPES`:

```typescript
export const INSPECTION_NODE_TYPES = new Set<string>([
  'binned_distribution',
  'data_preview',
  'outlier_monitor',
  'skewness_distribution',
  'dataset_profile',
  'your_new_inspection_node', // ← Add here
]);
```

**Examples:**
- Statistical viewers: `correlation_matrix`, `statistical_summary`
- Data quality monitors: `data_quality_report`
- Visualization nodes: `distribution_viewer`, `scatter_plot_preview`

## How to Decide: Transformation vs Inspection?

### Transformation Node → Do NOTHING
- **Modifies data**: Adds/removes/changes columns or rows
- **Creates new features**: Generates derived columns
- **Filters data**: Removes rows based on conditions
- **Transforms values**: Changes existing data (scaling, encoding, etc.)
- **Prepares for ML**: Feature engineering, train/test split

### Inspection Node → Add to Set
- **Only views**: Shows statistics, charts, distributions
- **No modifications**: Data passes through unchanged
- **Diagnostic**: Helps users understand data quality/patterns
- **Monitoring**: Checks for outliers, missing values, etc.

## Visual Status Indicators

Both node types show the same visual controls, but:

### Transformation Nodes
- ✅ Show spinner after save (background execution running)
- ✅ Show green checkmark when ready
- ✅ Show red error if failed
- ✅ Reset clears status

### Inspection Nodes
- ❌ No spinner (no background execution)
- ❌ No status indicators
- ✅ Only show settings and remove buttons

## Complete Node Addition Checklist

### Backend Setup
1. ✅ Create transformation logic in `core/`
2. ✅ Add API endpoint (if needed)
3. ✅ Update `core/feature_engineering/node_catalog.json` with new node definition
4. ✅ Add default config template

### Frontend Setup
1. ✅ Node appears in catalog automatically from `core/feature_engineering/node_catalog.json`
2. ✅ Create config UI in NodeSettingsModal (if needed)
3. ✅ Add `useCatalogFlags` entry (if special UI needed)
4. ✅ **ONLY IF INSPECTION NODE**: Add to `INSPECTION_NODE_TYPES` in `catalogTypes.ts`

### Background Execution
- ✅ **Automatic for transformation nodes** - Nothing to do!
- ✅ **Disabled for inspection nodes** - Add to `INSPECTION_NODE_TYPES`

## Examples

### Example 1: Adding "Text Vectorization" (Transformation)

**Backend** (`core/feature_engineering/node_catalog.json`):
```json
{
  "id": "text_vectorization",
  "label": "Text Vectorization",
  "category": "Feature Engineering",
  "description": "Convert text to numerical vectors",
  "default_config": {
    "method": "tfidf",
    "max_features": 100
  }
}
```

**Frontend**: 
- ❌ NO changes to `catalogTypes.ts` needed
- ✅ Background execution automatic
- ✅ Status indicators work automatically

### Example 2: Adding "Correlation Matrix" (Inspection)

**Backend** (`core/feature_engineering/node_catalog.json`):
```json
{
  "id": "correlation_matrix",
  "label": "Correlation Matrix",
  "category": "Data Quality",
  "description": "View correlation between features",
  "default_config": {
    "method": "pearson"
  }
}
```

**Frontend** (`catalogTypes.ts`):
```typescript
export const INSPECTION_NODE_TYPES = new Set<string>([
  'binned_distribution',
  'data_preview',
  'outlier_monitor',
  'skewness_distribution',
  'dataset_profile',
  'correlation_matrix', // ← ADD THIS
]);
```

- ✅ No background execution (it's inspection)
- ✅ No status indicators shown

## Testing New Nodes

### For Transformation Nodes:
1. Add node to canvas
2. Configure settings
3. Click "Save Changes"
4. ✅ Should see **blue spinner** appear immediately
5. ✅ Should turn to **green checkmark** when complete
6. Click "Reset"
7. ✅ Status should clear (no indicators)
8. Click "Reset All"
9. ✅ All status indicators should clear

### For Inspection Nodes:
1. Add node to canvas
2. Configure settings
3. Click "Save Changes"
4. ✅ Should see **no spinner** (no background execution)
5. ✅ No status indicators at any time
6. Data should load only when opening node

## Summary Table

| Node Type | Add to INSPECTION_NODE_TYPES? | Background Execution? | Status Indicators? |
|-----------|-------------------------------|----------------------|-------------------|
| Transformation (default) | ❌ No | ✅ Yes | ✅ Yes |
| Inspection | ✅ Yes | ❌ No | ❌ No |

## File Reference

- **Classification**: `frontend/feature-canvas/src/components/node-settings/utils/catalogTypes.ts`
- **Background Trigger**: `frontend/feature-canvas/src/components/NodeSettingsModal.tsx` (line ~4344)
- **Visual Indicators**: `frontend/feature-canvas/src/App.tsx` (FeatureCanvasNode component)
- **Flags Hook**: `frontend/feature-canvas/src/components/node-settings/hooks/useCatalogFlags.ts`

## TL;DR

**New transformation node?** → Do nothing, it just works! ✅  
**New inspection node?** → Add to `INSPECTION_NODE_TYPES` in `catalogTypes.ts` ✅
