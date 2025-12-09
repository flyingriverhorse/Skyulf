import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { DataSources } from './pages/DataSources';
import { CanvasPage } from './pages/CanvasPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Main App Layout (with Sidebar) */}
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="data" element={<DataSources />} />
          <Route path="deployments" element={<div className="p-8">Deployments (Coming Soon)</div>} />
        </Route>

        {/* Full Screen Canvas Layout (No Sidebar) */}
        <Route path="/canvas" element={<CanvasPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
