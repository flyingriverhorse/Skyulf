import { ErrorState } from './ErrorState';

export default { title: 'Shared / ErrorState' };

export const Basic = () => <ErrorState error="Something went wrong loading the dataset." />;

export const WithRetry = () => (
  <ErrorState
    error="Network request failed."
    onRetry={() => alert('retry clicked')}
  />
);
