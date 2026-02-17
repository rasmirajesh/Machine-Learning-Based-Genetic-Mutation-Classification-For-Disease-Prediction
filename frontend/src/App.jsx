import { Routes, Route } from "react-router-dom";
import Home from "./HomePage.jsx";   // 👈 FIXED
import Predict from "./Predict.jsx";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/predict" element={<Predict />} />
    </Routes>
  );
}

export default App;
