export const toHandleKey = (nodeId: string, handleId?: string | null): string | null => {
  if (!handleId || typeof handleId !== 'string') {
    return null;
  }
  const prefix = `${nodeId}-`;
  if (handleId.startsWith(prefix)) {
    return handleId.slice(prefix.length);
  }
  const parts = handleId.split('-');
  if (parts.length >= 2) {
    return parts.slice(-2).join('-');
  }
  return handleId;
};
