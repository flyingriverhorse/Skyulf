# Quick Start: Implementing Automatic Split Detection

## For Developers: 3-Step Implementation

### Step 1: Import the Split Handler (30 seconds)

Add to your `routes.py`:

```python
from core.feature_engineering.split_handler import (
    detect_splits,
    log_split_processing,
    remove_split_column,
)
```

### Step 2: Add Split Detection to Pipeline Loop (2 minutes)

In your pipeline execution function, add split detection:

```python
def execute_pipeline(nodes, edges, working_frame, pipeline_id=None):
    applied_steps = []
    
    for node in nodes:
        node_id = node.get("id")
        catalog_type = node.get("data", {}).get("catalogType", "")
        
        # ADD THIS: Detect and log splits
        split_info = detect_splits(working_frame)
        log_split_processing(node_id, catalog_type, split_info)
        
        # Your existing node processing code stays the same!
        if catalog_type == "scale_numeric_features":
            working_frame, summary, signal = apply_scale_numeric_features(
                working_frame, node, pipeline_id=pipeline_id
            )
        # ... rest of your nodes
        
        applied_steps.append(summary)
    
    # ADD THIS: Remove internal split column before returning
    final_frame = remove_split_column(working_frame)
    
    return final_frame, applied_steps
```

### Step 3: Enjoy Automatic Split Handling (0 minutes)

That's it! Your nodes now automatically:
- ✅ Detect train/test/validation splits
- ✅ Fit transformers on train only
- ✅ Transform all splits appropriately
- ✅ Process filters independently per split
- ✅ Apply resampling to train only
- ✅ Log all split operations
- ✅ Prevent data leakage

## For Users: How It Works

### Creating Splits

Use the **Train/Test Split** node in your workflow:

1. Add **Train/Test Split** node to canvas
2. Configure split ratios:
   - Test size: 0.2 (20%)
   - Validation size: 0.1 (10%) - optional
   - Random state: 42
3. Connect to your dataset

### What Happens Next

**All downstream nodes automatically detect and handle splits:**

#### Transformers (Scalers, Encoders, Imputers)
- Fit on **training data only** ← Prevents data leakage!
- Transform **all splits** (train, test, validation)
- Example: StandardScaler learns mean/std from train, applies to all

#### Filters (Remove Duplicates, Drop Nulls, Outlier Removal)
- Process **each split independently**
- Train duplicates don't affect test
- Example: Remove duplicates separately in train and test

#### Resampling (SMOTE, Undersampling)
- Apply **only to training data** ← Prevents data leakage!
- Test and validation unchanged
- Example: SMOTE creates synthetic samples in train only

#### Models (Training, Prediction)
- Train on **training data**
- Predict on **test/validation data**
- Example: RandomForest fits on train, predicts on test

### Visual Example

```
Dataset (1000 rows)
    ↓
Train/Test Split (70/20/10)
    ↓
├─ Train: 700 rows
├─ Test: 200 rows
└─ Validation: 100 rows
    ↓
StandardScaler Node
    ├─ FIT on train (700 rows) → learns μ=50, σ=10
    ├─ TRANSFORM train (700 rows) using μ=50, σ=10
    ├─ TRANSFORM test (200 rows) using μ=50, σ=10  ← No leakage!
    └─ TRANSFORM validation (100 rows) using μ=50, σ=10  ← No leakage!
    ↓
Remove Duplicates Node
    ├─ Process train (700 → 680 rows)
    ├─ Process test (200 → 195 rows)
    └─ Process validation (100 → 98 rows)
    ↓
SMOTE Oversampling Node
    ├─ Apply to train (680 → 900 rows)  ← Balanced classes
    ├─ Test unchanged (195 rows)  ← No synthetic samples!
    └─ Validation unchanged (98 rows)  ← No synthetic samples!
    ↓
Model Training Node
    ├─ Train on train (900 rows)
    ├─ Predict on test (195 rows)
    └─ Predict on validation (98 rows)
```

## Benefits for You

### Before (Manual Split Handling)
```
❌ Needed to remember which nodes to apply to which split
❌ Easy to accidentally cause data leakage
❌ Had to manually configure each node
❌ Complex to track what happened where
❌ Errors were common
```

### After (Automatic Split Detection)
```
✅ Splits detected automatically
✅ Data leakage prevented automatically
✅ No manual configuration needed
✅ Clear logs show what happened
✅ Just works!
```

## Common Workflows

### Workflow 1: Basic Classification
```
1. Upload Dataset
2. Train/Test Split (80/20)
3. StandardScaler → Auto: fit on train, transform both
4. One-Hot Encoding → Auto: fit on train, transform both
5. Train Model → Auto: fit on train, predict on test
```

### Workflow 2: Imbalanced Classes
```
1. Upload Dataset
2. Train/Test Split (70/30)
3. SMOTE Oversampling → Auto: only on train!
4. StandardScaler → Auto: fit on train, transform both
5. Train Model → Auto: fit on train, predict on test
```

### Workflow 3: With Validation Set
```
1. Upload Dataset
2. Train/Test/Validation Split (70/20/10)
3. StandardScaler → Auto: fit on train, transform all 3
4. Feature Engineering → Auto: fit on train, transform all 3
5. Train Model → Auto: fit on train, predict on test + validation
```

## Monitoring

### Check Split Information

The system automatically logs:
```
INFO: Node scaler-1 (scale_numeric_features): processing with splits 
      [train=700, test=200, validation=100] - Category: transformer
INFO: ✓ Fitted scaler on 700 train rows
INFO: ✓ Transformed 1000 total rows (train + test + validation)
```

### In the UI

Split information is shown:
- Node status shows "Processing train split..." 
- Results show split-aware summaries
- Transformer Audit Report shows fit/transform per split

## Troubleshooting

### Q: My node isn't detecting splits
**A:** Check that:
1. Train/Test Split node is connected upstream
2. Pipeline execution is sequential
3. Logs show "No splits detected" → Check connections

### Q: How do I know if it's working?
**A:** Look for logs like:
- "processing with splits [train=X, test=Y]"
- "Fitted on X train rows"
- "Transformed Y total rows"

### Q: Can I disable it for a specific node?
**A:** Yes, the node will process normally if no splits are detected.
Just don't connect it downstream of a split node.

### Q: What if I want manual control?
**A:** You can still manually configure nodes. The automatic detection
enhances the system but doesn't restrict manual control.

## Next Steps

1. ✅ Try the Train/Test Split node
2. ✅ Add a StandardScaler downstream
3. ✅ Check the logs to see automatic split handling
4. ✅ Review the Transformer Audit Report
5. ✅ Build your ML pipeline!

## Summary

**Old Way:**
- 🤔 "Did I apply the scaler correctly?"
- 🤔 "Is there data leakage?"
- 🤔 "Which split is this?"
- 😰 Manual configuration everywhere

**New Way:**
- ✅ Automatic split detection
- ✅ Zero data leakage
- ✅ Clear split tracking
- 😊 Just works!

**You now have enterprise-grade split handling with zero extra effort!** 🚀
