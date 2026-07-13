import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import Home from "./pages/Home";
import LabDashboard from "./pages/LabDashboard";
import LabDatasets from "./pages/LabDatasets";
import LabDiff from "./pages/LabDiff";
import LabNewRun from "./pages/LabNewRun";
import LabQA from "./pages/LabQA";
import LabRunViewer from "./pages/LabRunViewer";
import Report from "./pages/Report";

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<Home />} />
        <Route path="/report/:runId" element={<Report />} />
        <Route path="/lab" element={<LabDashboard />} />
        <Route path="/lab/new" element={<LabNewRun />} />
        <Route path="/lab/runs/:runId" element={<LabRunViewer />} />
        <Route path="/lab/diff/:a/:b" element={<LabDiff />} />
        <Route path="/lab/qa" element={<LabQA />} />
        <Route path="/lab/datasets" element={<LabDatasets />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
