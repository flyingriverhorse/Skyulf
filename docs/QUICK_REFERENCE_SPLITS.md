# 🎯 QUICK REFERENCE: Automatic Split Detection

## For Users

### How to Use
1. Add **Train/Test Split** node to your pipeline
2. Connect downstream nodes
3. Execute pipeline
4. ✅ Splits handled automatically!

### What Happens Automatically

| Node Type | Automatic Behavior |
|-----------|-------------------|
| **Scalers** (StandardScaler, MinMaxScaler) | Fit on train → Transform all |
| **Encoders** (OneHot, Label, Ordinal) | Fit on train → Transform all |
| **Imputers** (Simple, Advanced) | Fit on train → Transform all |
| **Deduplication** | Process each split independently |
| **Outlier Removal** | Process each split independently |
| **SMOTE/Resampling** | Only on train (test unchanged) |
| **Models** | Train on train → Predict on test |

### Benefits
- ✅ No manual configuration
- ✅ No data leakage
- ✅ No adjusting needed
- ✅ Professional ML workflow

---

## For Developers

### Integration Points

**1. Import (routes.py line ~100):**
```python
from .split_handler import (
    detect_splits, log_split_processing, 
    remove_split_column, get_node_category
)
```

**2. Detect (routes.py line ~1268):**
```python
split_info = detect_splits(working_frame)
log_split_processing(node_id, catalog_type, split_info)
```

**3. Cleanup (routes.py line ~1512):**
```python
working_frame = remove_split_column(working_frame)
```

### Node Categories

```python
# In split_handler.py
NODE_CATEGORY_MAP = {
    "scale_numeric_features": NodeCategory.TRANSFORMER,
    "remove_duplicates": NodeCategory.FILTER,
    "class_oversampling": NodeCategory.TRANSFORMER,  # Special
    "train_test_split": NodeCategory.SPLITTER,
}
```

### Adding New Nodes

```python
# Just add to NODE_CATEGORY_MAP
NODE_CATEGORY_MAP["my_new_node"] = NodeCategory.TRANSFORMER
```

---

## Log Examples

### With Splits
```
INFO: Node split-1: Created splits [train=800, test=200]
INFO: Node scaler-1 (scale_numeric_features): processing with splits 
      [train=800, test=200] - Category: transformer
INFO: ✓ Fitted on 800 train rows, transformed 1000 total rows
```

### Without Splits
```
INFO: Node scaler-1 (scale_numeric_features): processing without splits
INFO: ✓ Scaled 1000 rows
```

---

## Common Patterns

### Pattern 1: Basic Split
```
Dataset → Train/Test Split (80/20) → StandardScaler → Model
```
Result: Scaler fits on 80%, transforms 100%

### Pattern 2: With Validation
```
Dataset → Train/Test/Val Split (70/20/10) → Scaler → Model
```
Result: Scaler fits on 70%, transforms 100%

### Pattern 3: Imbalanced Data
```
Dataset → Split → SMOTE → Scaler → Model
```
Result: SMOTE on train only, Scaler fits on resampled train

---

## Testing

### Quick Test
```bash
python test_split_integration.py
```

### In Pipeline
1. Create pipeline with split
2. Check logs for "processing with splits"
3. Verify fit/transform behavior

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No split detection | Check split node is connected |
| Wrong behavior | Check NODE_CATEGORY_MAP |
| No logs | Set logging to INFO level |

---

## Files Reference

| File | Purpose |
|------|---------|
| `split_handler.py` | Core system |
| `routes.py` | Integration |
| `AUTOMATIC_SPLIT_DETECTION.md` | Full docs |
| `QUICK_START_SPLIT_DETECTION.md` | User guide |
| `SOLUTION_SUMMARY.md` | Overview |

---

## Key Functions

```python
detect_splits(df) → SplitInfo
log_split_processing(node_id, catalog_type, split_info)
remove_split_column(df) → df
get_node_category(catalog_type) → NodeCategory
```

---

## Summary

**Before:** Manual split management, constant adjusting  
**After:** Automatic detection, zero configuration  
**Result:** Professional ML pipeline with data leakage prevention 🎉
