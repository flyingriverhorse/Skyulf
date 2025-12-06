import { registry } from './NodeRegistry';
import { DebugNode } from '../../modules/nodes/base/DebugNode';
import { DatasetNode } from '../../modules/nodes/data/DatasetNode';
import { DropColumnsNode } from '../../modules/nodes/processing/DropColumnsNode';
import { ImputationNode } from '../../modules/nodes/processing/ImputationNode';
import { ScalingNode } from '../../modules/nodes/processing/ScalingNode';
import { ModelTrainingNode } from '../../modules/nodes/modeling/ModelTrainingNode';
import { EvaluationNode } from '../../modules/nodes/modeling/EvaluationNode';
import { FeatureSelectionNode } from '../../modules/nodes/processing/FeatureSelectionNode';
import { TrainTestSplitNode } from '../../modules/nodes/modeling/TrainTestSplitNode';
import { FeatureTargetSplitNode } from '../../modules/nodes/modeling/FeatureTargetSplitNode';
import { DropRowsNode } from '../../modules/nodes/processing/DropRowsNode';
import { DeduplicationNode } from '../../modules/nodes/processing/DeduplicationNode';
import { CastTypeNode } from '../../modules/nodes/processing/CastTypeNode';
import { MissingIndicatorNode } from '../../modules/nodes/processing/MissingIndicatorNode';
import { EncodingNode } from '../../modules/nodes/processing/EncodingNode';
import { OutlierNode } from '../../modules/nodes/processing/OutlierNode';

export const initializeRegistry = () => {
  registry.register(DebugNode);
  registry.register(DatasetNode);
  registry.register(DropColumnsNode);
  registry.register(ImputationNode);
  registry.register(ScalingNode);
  registry.register(OutlierNode);
  registry.register(EncodingNode);
  registry.register(FeatureSelectionNode);
  registry.register(TrainTestSplitNode);
  registry.register(FeatureTargetSplitNode);
  registry.register(DropRowsNode);
  registry.register(DeduplicationNode);
  registry.register(CastTypeNode);
  registry.register(MissingIndicatorNode);
  registry.register(ModelTrainingNode);
  registry.register(EvaluationNode);
  
  console.log('[Registry] Initialization complete. Total nodes:', registry.getAll().length);
};
