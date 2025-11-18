import { useMemo } from 'react';
import type { ReactElement } from 'react';
import type { FeatureNodeCatalogEntry } from '../../api';
import { FeatureCanvasSidebar } from '../../components/FeatureCanvasSidebar';

 type UseCanvasSidebarOptions = {
   nodeCatalog: FeatureNodeCatalogEntry[];
   isCatalogLoading: boolean;
   catalogErrorMessage: string | null;
   handleAddNode: (catalogNode: FeatureNodeCatalogEntry) => void;
 };
 
 type UseCanvasSidebarResult = {
   sidebarContent: ReactElement;
 };
 
 export const useCanvasSidebar = ({
   nodeCatalog,
   isCatalogLoading,
   catalogErrorMessage,
   handleAddNode,
 }: UseCanvasSidebarOptions): UseCanvasSidebarResult => {
   const sidebarContent = useMemo(() => {
     if (isCatalogLoading) {
       return <p className="text-muted">Loading node catalog…</p>;
     }
     if (catalogErrorMessage) {
       return <p className="text-danger">{catalogErrorMessage}</p>;
     }
     if (!nodeCatalog.length) {
       return <p className="text-muted">Node catalog unavailable. Define nodes in the backend to continue.</p>;
     }
     return <FeatureCanvasSidebar nodes={nodeCatalog} onAddNode={handleAddNode} />;
   }, [catalogErrorMessage, handleAddNode, isCatalogLoading, nodeCatalog]);
 
   return {
     sidebarContent,
   };
 };
