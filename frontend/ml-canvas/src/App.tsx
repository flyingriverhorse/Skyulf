import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { DataSources } from './pages/DataSources';
import { CanvasPage } from './pages/CanvasPage';
import { ModelRegistry } from './pages/ModelRegistry';
import { DeploymentsPage } from './components/pages/DeploymentsPage';
import { EDAPage } from './pages/EDAPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Main App Layout (with Sidebar) */}
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="data" element={<DataSources />} />
          <Route path="eda" element={<EDAPage />} />
          <Route path="canvas" element={<CanvasPage />} />
          <Route path="registry" element={<ModelRegistry />} />
          <Route path="deployments" element={<DeploymentsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
