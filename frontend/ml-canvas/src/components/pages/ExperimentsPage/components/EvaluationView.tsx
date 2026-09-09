import type { FC } from 'react';
import type { EvaluationViewProps } from './evaluationView/types';
import { useThresholdMutation } from './evaluationView/useThresholdMutation';
import { useEvaluationPresentation } from './evaluationView/useEvaluationPresentation';
import { EvaluationRunSelector } from './evaluationView/EvaluationRunSelector';
import { EvaluationBody } from './evaluationView/EvaluationBody';

/** Keep evaluation mutation state alive across loading, errors and threshold tabs. */
export const EvaluationView: FC<EvaluationViewProps> = (props) => {
  const mutation = useThresholdMutation();
  const presentation = useEvaluationPresentation(props);
  return <div className="space-y-6">
    <EvaluationRunSelector {...props} eligibleRunLabels={presentation.eligibleRunLabels} />
    <EvaluationBody {...props} {...presentation} {...mutation} />
  </div>;
};
