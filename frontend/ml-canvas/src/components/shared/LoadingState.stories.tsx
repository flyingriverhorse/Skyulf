import { LoadingState } from './LoadingState';

export default { title: 'Shared / LoadingState' };

export const Default = () => <LoadingState />;
export const CustomMessage = () => <LoadingState message="Crunching numbers..." />;
