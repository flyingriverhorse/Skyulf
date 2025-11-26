import { useEffect, useRef, type Dispatch, type SetStateAction } from 'react';
import { type CatalogFlagMap } from './useCatalogFlags';

interface UseTargetEncodingDefaultsArgs {
  catalogFlags: CatalogFlagMap;
  enableGlobalFallbackDefault: boolean;
  encodeMissing: any;
  handleUnknown: any;
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
  nodeChangeVersion: number;
}

export const useTargetEncodingDefaults = ({
  catalogFlags,
  enableGlobalFallbackDefault,
  encodeMissing,
  handleUnknown,
  setConfigState,
  nodeChangeVersion,
}: UseTargetEncodingDefaultsArgs) => {
  const { isTargetEncodingNode } = catalogFlags;
  const fallbackAppliedRef = useRef(false);

  useEffect(() => {
    fallbackAppliedRef.current = false;
  }, [nodeChangeVersion]);

  useEffect(() => {
    if (!isTargetEncodingNode || !enableGlobalFallbackDefault) {
      return;
    }
    if (fallbackAppliedRef.current) {
      return;
    }
    const encodeMissingActive = typeof encodeMissing === 'boolean' ? encodeMissing : null;
    const currentHandleUnknown = typeof handleUnknown === 'string'
      ? handleUnknown.trim().toLowerCase()
      : '';

    if (encodeMissingActive === true && currentHandleUnknown === 'global_mean') {
      fallbackAppliedRef.current = true;
      return;
    }

    setConfigState((previous) => {
      const previousEncodeMissing = typeof previous?.encode_missing === 'boolean' ? previous.encode_missing : null;
      const previousHandleUnknown = typeof previous?.handle_unknown === 'string'
        ? previous.handle_unknown.trim().toLowerCase()
        : '';

      if (previousEncodeMissing === true && previousHandleUnknown === 'global_mean') {
        fallbackAppliedRef.current = true;
        return previous;
      }

      const next: Record<string, any> = { ...previous };

      if (previousEncodeMissing !== true) {
        next.encode_missing = true;
      }

      if (previousHandleUnknown !== 'global_mean') {
        next.handle_unknown = 'global_mean';
      }

      fallbackAppliedRef.current = true;
      return next;
    });
  }, [
    encodeMissing,
    handleUnknown,
    isTargetEncodingNode,
    enableGlobalFallbackDefault,
    setConfigState,
  ]);
};
