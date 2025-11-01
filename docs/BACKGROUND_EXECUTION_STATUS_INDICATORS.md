# Background Execution Status Indicators

## Overview
This document describes the visual status indicators added to nodes in the feature canvas to show the progress and completion of background full dataset execution jobs.

## Visual Indicators

When you save changes to a node's configuration, the system triggers a background job to pre-load the full dataset for that node. Visual indicators on the node show the status of this background process:

### 1. **Loading Spinner** (Blue)
- **Appears:** Immediately after clicking "Save Changes" on a node
- **Meaning:** Background full dataset execution is in progress
- **Visual:** Animated spinning circle in blue color
- **Location:** Top-right area of the node, next to other controls

### 2. **Success Checkmark** (Green)
- **Appears:** When the background execution completes successfully
- **Meaning:** Full dataset is ready and cached for this node
- **Visual:** Green checkmark (✓) with green border and background
- **Location:** Same position as the loading spinner

### 3. **Error Indicator** (Red)
- **Appears:** If the background execution fails
- **Meaning:** Background execution encountered an error
- **Visual:** Red exclamation mark (!) with red border and background
- **Location:** Same position as other status indicators

## Technical Implementation

### Frontend Components

#### 1. **FeatureNodeData Type** (`App.tsx`)
```typescript
type FeatureNodeData = {
  label?: string;
  description?: string;
  catalogType?: string;
  config?: Record<string, any>;
  isConfigured?: boolean;
  backgroundExecutionStatus?: 'idle' | 'loading' | 'success' | 'error';
};
```

#### 2. **FeatureCanvasNode Component** (`App.tsx`)
The node component now displays status indicators based on `backgroundExecutionStatus`:

```typescript
const backgroundStatus = data?.backgroundExecutionStatus ?? 'idle';

// Loading spinner
{backgroundStatus === 'loading' && (
  <span className="feature-node__status-indicator feature-node__status-indicator--loading">
    <svg className="feature-node__spinner" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10" ... />
    </svg>
  </span>
)}

// Success checkmark
{backgroundStatus === 'success' && (
  <span className="feature-node__status-indicator feature-node__status-indicator--success">
    ✓
  </span>
)}

// Error indicator
{backgroundStatus === 'error' && (
  <span className="feature-node__status-indicator feature-node__status-indicator--error">
    !
  </span>
)}
```

#### 3. **CSS Styles** (`styles.css`)

**Base Status Indicator:**
```css
.feature-node__status-indicator {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 999px;
  font-size: 0.9rem;
  line-height: 1;
}
```

**Loading State:**
```css
.feature-node__status-indicator--loading {
  color: rgba(59, 130, 246, 0.9); /* Blue */
}

.feature-node__spinner {
  width: 18px;
  height: 18px;
  animation: feature-node-spin 1s linear infinite;
}

@keyframes feature-node-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
```

**Success State:**
```css
.feature-node__status-indicator--success {
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid rgba(16, 185, 129, 0.4);
  color: rgba(16, 185, 129, 0.95); /* Green */
  font-weight: 700;
  font-size: 1.1rem;
}
```

**Error State:**
```css
.feature-node__status-indicator--error {
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.4);
  color: rgba(239, 68, 68, 0.95); /* Red */
  font-weight: 700;
  font-size: 1.1rem;
}
```

### Data Flow

#### 1. **Node Update Callbacks** (`App.tsx`)

**handleUpdateNodeData** - New callback for updating non-config node data:
```typescript
const handleUpdateNodeData = useCallback(
  (nodeId: string, dataUpdates: Partial<FeatureNodeData>) => {
    setNodes((current) =>
      current.map((node) => {
        if (node.id !== nodeId) return node;
        
        const baseData = {
          ...(node.data ?? {}),
          ...dataUpdates,
        };

        return registerNodeInteractions({
          ...node,
          data: baseData,
        });
      })
    );
  },
  [registerNodeInteractions, setNodes]
);
```

#### 2. **NodeSettingsModal Integration** (`NodeSettingsModal.tsx`)

**Props Extension:**
```typescript
type NodeSettingsModalProps = {
  // ... existing props
  onUpdateNodeData?: (nodeId: string, dataUpdates: Record<string, any>) => void;
};
```

**handleSave Modification:**
```typescript
const handleSave = useCallback(() => {
  // ... save config logic
  
  onUpdateConfig(node.id, payload);
  onClose();
  
  // Trigger background execution for non-inspection nodes
  if (sourceId && graphSnapshot && !isInspectionNode) {
    // Set loading status immediately
    if (onUpdateNodeData) {
      onUpdateNodeData(node.id, { backgroundExecutionStatus: 'loading' });
    }
    
    triggerFullDatasetExecution({
      dataset_source_id: sourceId,
      graph: {
        nodes: graphSnapshot.nodes || [],
        edges: graphSnapshot.edges || [],
      },
      target_node_id: node.id,
    })
      .then(() => {
        // Update to success when complete
        if (onUpdateNodeData) {
          onUpdateNodeData(node.id, { backgroundExecutionStatus: 'success' });
        }
      })
      .catch((error) => {
        // Update to error on failure
        if (onUpdateNodeData) {
          onUpdateNodeData(node.id, { backgroundExecutionStatus: 'error' });
        }
        console.warn('Background full dataset execution failed:', error);
      });
  }
}, [/* deps */]);
```

## User Experience Flow

1. **User configures a node** and clicks "Save Changes"
2. **Modal closes**, changes are saved
3. **Spinner appears** on the node immediately (status: 'loading')
4. **Background job runs** to pre-load full dataset
5. **When complete:**
   - **Success:** Green checkmark appears (status: 'success')
   - **Error:** Red exclamation mark appears (status: 'error')
6. **User knows** the full dataset is ready without opening the node

## Node Types Affected

The background execution (and status indicators) apply to **transformation nodes only**, not inspection nodes:

### Transformation Nodes (28 nodes - show status):
- drop_columns, drop_duplicates, drop_nulls, fill_missing
- filter_rows, sort_rows, sample_rows, rename_columns
- binning, one_hot_encoding, label_encoding, target_encoding
- ordinal_encoding, frequency_encoding, binary_encoding
- scaling, log_transform, box_cox_transform, yeo_johnson_transform
- feature_interaction, polynomial_features, pca, aggregation
- date_features, text_features, custom_transform, train_test_split

### Inspection Nodes (4 nodes - no status indicators):
- binned_distribution, data_preview, outlier_monitor, skewness_distribution, dataset_profile

**Note:** Although `data_preview` (Dataset Snapshot) has a "Refresh Full Dataset" button and could benefit from background execution, it is currently classified as an inspection node and does not trigger background execution.

## Benefits

1. **Visual Feedback:** Users can see when background jobs are running
2. **Status Awareness:** Users know when full dataset is ready
3. **Error Notification:** Users are alerted to background failures
4. **Non-Blocking UX:** Background execution doesn't freeze the UI
5. **Smart Pre-loading:** Full dataset ready before user needs it

## Future Enhancements

Potential improvements:
- Click on error indicator to show error details
- Progress percentage for long-running jobs
- Ability to cancel background execution
- Retry failed executions
- Status persistence across page reloads
- Status indicators in node list/tree view
