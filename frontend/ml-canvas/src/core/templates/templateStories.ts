/** Outcome-first copy for the five curated starter workflows. */
export const templateStories: Record<string, {
  title: string; example: string; needs: string; outcome: string; steps: string[];
}> = {
  tabular_classification: {
    title: 'Predict a category', example: 'Will a customer stay or leave?',
    needs: 'A table with a known outcome for each training row.',
    outcome: 'Compare predicted categories with known outcomes in Experiments.',
    steps: ['Select data', 'Split', 'Prepare features', 'Train classifier'],
  },
  tabular_regression: {
    title: 'Estimate a value', example: 'How much could a home sell for?',
    needs: 'Numeric features and a numeric target, such as price or demand.',
    outcome: 'Inspect prediction errors and see how closely estimates match real values.',
    steps: ['Select data', 'Split', 'Clean & scale', 'Train regressor'],
  },
  text_classification: {
    title: 'Make sense of text', example: 'Sort messages into useful categories.',
    needs: 'A text column and a separate column containing its category.',
    outcome: 'Evaluate a text classifier built with TF-IDF and logistic regression.',
    steps: ['Select text', 'Split', 'Clean & vectorize', 'Classify'],
  },
  customer_segmentation: {
    title: 'Discover customer groups', example: 'Find customers with similar behaviour.',
    needs: 'Numeric features describing each customer. No target column needed.',
    outcome: 'Explore the resulting groups in the Experiments Segmentation tab.',
    steps: ['Select data', 'Fill missing values', 'Scale', 'Find groups'],
  },
  ensemble_classification: {
    title: 'Combine model predictions', example: 'Try a voting ensemble for your classification task.',
    needs: 'A table with known categories. Review the base models before training.',
    outcome: 'Evaluate combined predictions; compare with a single-model run to check whether they help.',
    steps: ['Select data', 'Split', 'Prepare features', 'Voting ensemble'],
  },
};
