import { Routes, Route } from "react-router-dom";
import Home from "./pages/HomePage.jsx";
import Predict from "./pages/Predict.jsx";
import Explainability from "./pages/Explainability.jsx";
import ModelComparison from "./pages/ModelComparison.jsx";
import About from "./pages/About.jsx";
import Navbar from "./components/Navbar.jsx";

function App() {
  return (
    <>
      <Navbar />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/explainability" element={<Explainability />} />
        <Route path="/model-comparison" element={<ModelComparison />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </>
  );
}

export default App;