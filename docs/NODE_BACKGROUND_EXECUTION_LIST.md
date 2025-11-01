# Complete List: Which Nodes Trigger Background Full Dataset Execution

## ✅ TRIGGERS Background Full Dataset (Transformation Nodes)

These nodes **WILL trigger** background full dataset execution when you click "Save Changes":

### Data Cleaning & Quality (7 nodes)
1. **drop_missing_columns** - Drop columns with too many missing values
2. **drop_missing_rows** - Drop rows with missing values
3. **remove_duplicates** - Remove duplicate rows
4. **cast_column_types** - Change column data types
5. **imputation_methods** - Fill missing values with strategies
6. **missing_value_indicator** - Create binary indicators for missing values
7. **outlier_removal** - Remove statistical outliers

### Text Processing & Cleanup (7 nodes)
8. **trim_whitespace** - Remove leading/trailing spaces
9. **normalize_text_case** - Standardize text to upper/lower/title case
10. **replace_aliases_typos** - Replace text aliases and fix typos
11. **standardize_date_formats** - Standardize date formats
12. **remove_special_characters** - Remove special characters from text
13. **replace_invalid_values** - Replace invalid/malformed values
14. **regex_replace_fix** - Regex-based text replacement

### Encoding & Categorical (6 nodes)
15. **label_encoding** - Encode categorical as integers
16. **target_encoding** - Encode based on target variable
17. **hash_encoding** - Hash-based encoding for high cardinality
18. **ordinal_encoding** - Ordered categorical encoding
19. **dummy_encoding** - Create dummy variables (drop first)
20. **one_hot_encoding** - Create one-hot encoded columns

### Feature Engineering (3 nodes)
21. **binning_discretization** - Create bins from numeric columns
22. **scale_numeric_features** - Scale/normalize numeric features
23. **skewness_transform** - Apply transformations to reduce skewness

### Sampling & Balancing (2 nodes)
24. **class_undersampling** - Balance classes by undersampling majority
25. **class_oversampling** - Balance classes by oversampling minority

### Modeling Preparation (2 nodes)
26. **feature_target_split** - Split features and target variable
27. **train_model_draft** - Train machine learning model

### Dataset Operations (1 node)
28. **data_preview** - View dataset snapshot with full dataset refresh capability

---

## ❌ DOES NOT TRIGGER (Inspection Nodes)

These nodes **WILL NOT trigger** background execution (they only view/inspect data):

### Inspection & Visualization (4 nodes)
1. **dataset_profile** - Generate lightweight dataset profile
2. **binned_distribution** - Visualize binned column distributions
3. **skewness_distribution** - Visualize skewness distributions
4. **outlier_monitor** - Monitor outliers (if exists)

---

## Summary Statistics

### Total Nodes: 32
- **✅ Transformation nodes (trigger):** 28 nodes
- **❌ Inspection nodes (no trigger):** 4 nodes

### Why This Split?

**Transformation nodes + Data Preview** = Modify or need full dataset
- Need full dataset to apply transformations correctly
- Data Preview can request full dataset via "Refresh Full Dataset" button
- Pre-loading ensures data ready for training/export or full preview
- Background execution prepares data while you work

**Pure Inspection nodes** = Only view dataset with samples
- Don't modify data, just visualize/inspect with samples
- No need for full dataset (samples are sufficient)
- Keep these fast for quick previews

---

## Code Logic

```typescript
// In NodeSettingsModal.tsx handleSave callback:

if (sourceId && graphSnapshot && !isInspectionNode) {
  // Triggers for all 27 transformation nodes ✅
  triggerFullDatasetExecution({...});
}

// isInspectionNode = true for these 4 nodes:
// - dataset_profile  
// - binned_distribution
// - skewness_distribution
// - outlier_monitor
//
// isInspectionNode = false for data_preview (needs full dataset capability)
```

---

## Examples by Category

### ✅ Example: Data Preview Node (TRIGGERS) 
```
User: Opens "Data Preview" node (Data Snapshot)
User: Views sample data (1000 rows)
User: Configures or just reviews
User: Clicks "Save Changes" or closes modal
  ↓
System: Saves any settings (instant)
System: Triggers background full dataset execution 🚀
System: Loads full dataset in background
  ↓
Result: Full dataset ready for "Refresh Full Dataset" button
Result: User can immediately see full data when clicking refresh ⚡
```

### ✅ Example: Encoding Node (TRIGGERS)
```
User: Opens "Label Encoding" node
User: Configures columns to encode
User: Clicks "Save Changes"
  ↓
System: Saves config (instant)
System: Triggers background full dataset execution 🚀
System: Loads 200k rows in background
System: Pre-processes encoding transformations
  ↓
Result: When user trains model, data already ready! ⚡
```

### ❌ Example: Dataset Profile Node (NO TRIGGER)
```
User: Opens "Dataset Profile" node
User: Views profile statistics
User: Clicks "Save" or closes modal
  ↓
System: Saves any settings (instant)
System: Does NOT trigger background execution ❌
  ↓
Result: Pure inspection node, no full dataset needed
```

---

## Quick Reference Table

| Node Category | Trigger? | Count | Why? |
|--------------|----------|-------|------|
| Data Cleaning | ✅ Yes | 7 | Modifies dataset |
| Text Processing | ✅ Yes | 7 | Modifies dataset |
| Encoding | ✅ Yes | 6 | Modifies dataset |
| Feature Engineering | ✅ Yes | 3 | Modifies dataset |
| Sampling/Balancing | ✅ Yes | 2 | Modifies dataset |
| Modeling Prep | ✅ Yes | 2 | Modifies dataset |
| Dataset Operations | ✅ Yes | 1 | Needs full dataset |
| **Inspection/Viz** | **❌ No** | **4** | **Only views samples** |

---

## How to Check in Your App

### In Browser Console:
```javascript
// When you save a transformation node:
// No output = successful background trigger

// When you save an inspection node:  
// No API call made (check Network tab)
```

### In Network Tab:
```
Transformation node save:
  POST /ml-workflow/api/pipelines/preview
  Payload: { ..., sample_size: 0 }
  
Inspection node save:
  (No POST request to preview endpoint)
```

---

## Performance Impact by Node Type

### Transformation Nodes (27 nodes)
- **Save action**: Instant (no blocking)
- **Background job**: Starts immediately
- **User impact**: Can continue working
- **Benefit**: Data pre-loaded for training

### Data Preview Node (1 node)
- **Save action**: Instant (no blocking)
- **Background job**: Starts immediately
- **User impact**: Can continue working
- **Benefit**: Full dataset ready for "Refresh Full Dataset" button

### Pure Inspection Nodes (4 nodes)  
- **Save action**: Instant
- **Background job**: None
- **User impact**: None (as intended)
- **Benefit**: Stays fast, no unnecessary processing

---

**Last Updated:** October 20, 2025  
**Code Reference:** `NodeSettingsModal.tsx` line ~4340  
**Condition:** `!isInspectionNode`  
**Inspection Types:** Defined in `catalogTypes.ts`
