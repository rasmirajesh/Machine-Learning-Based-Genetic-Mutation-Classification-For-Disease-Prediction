import { Link } from "react-router-dom";

export default function Predict() {
  return (
    <div className="min-h-screen bg-[#f5f7f9]">

      {/* Navbar */}
      <nav className="flex justify-between items-center px-10 py-4 bg-white shadow-sm">
        <h1 className="text-xl font-semibold text-[#1e3a5f]">
          GeneMutate AI
        </h1>

        <div className="space-x-6 text-gray-600 font-medium">
          <Link to="/" className="hover:text-[#1e3a5f]">Home</Link>
          <Link to="/predict" className="text-[#1e3a5f] font-semibold">
            Predict
          </Link>
          <span className="hover:text-[#1e3a5f] cursor-pointer">
            Explainability
          </span>
          <span className="hover:text-[#1e3a5f] cursor-pointer">
            Model Comparison
          </span>
          <span className="hover:text-[#1e3a5f] cursor-pointer">
            About
          </span>
        </div>
      </nav>

      {/* Page Content */}
      <div className="px-20 py-12">

        {/* Title */}
        <h2 className="text-3xl font-bold text-gray-800">
          Disease Prediction
        </h2>
        <p className="text-gray-500 mt-2">
          Enter genetic mutation features to predict the associated disease.
        </p>

        {/* Two Column Layout */}
        <div className="grid grid-cols-2 gap-8 mt-10">

          {/* LEFT CARD */}
          <div className="bg-white rounded-xl shadow-sm p-8">

            <h3 className="text-lg font-semibold text-gray-700 mb-6">
              Input Features
            </h3>

            {/* Risk Score */}
            <div className="mb-5">
              <label className="block text-sm font-medium text-gray-600 mb-2">
                Risk Score (0.0 – 1.0)
              </label>
              <input
                type="number"
                placeholder="e.g., 0.75"
                step="0.01"
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6c8ea3]"
              />
            </div>

            {/* Mutation Status */}
            <div className="mb-5">
              <label className="block text-sm font-medium text-gray-600 mb-2">
                Mutation Status
              </label>
              <select className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6c8ea3]">
                <option>Select mutation status</option>
                <option>Present</option>
                <option>Absent</option>
              </select>
            </div>

            {/* Clinical Significance */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-600 mb-2">
                Clinical Significance (1–5)
              </label>
              <select className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6c8ea3]">
                <option>Select significance level</option>
                <option>1 - Low</option>
                <option>2</option>
                <option>3</option>
                <option>4</option>
                <option>5 - High</option>
              </select>
            </div>

            {/* Buttons */}
            <div className="flex gap-4">
              <button className="w-full bg-[#6c8ea3] text-white py-2 rounded-lg hover:opacity-90 transition">
                Predict Disease
              </button>

              <button className="border border-gray-300 px-6 py-2 rounded-lg text-gray-600 hover:bg-gray-100 transition">
                Reset
              </button>
            </div>
          </div>

          {/* RIGHT CARD */}
          <div className="bg-white rounded-xl shadow-sm p-8 flex flex-col justify-center items-center text-center">

            <div className="text-5xl text-gray-300 mb-4">
              🧪
            </div>

            <h4 className="text-lg font-semibold text-gray-700">
              No prediction yet
            </h4>
            <p className="text-gray-500 mt-2">
              Fill in the input features and click "Predict Disease"
            </p>

          </div>

        </div>
      </div>

      {/* Footer */}
      <footer className="text-center text-sm text-gray-500 py-6 border-t mt-12">
        © 2026 Genetic Mutation Detection & Disease Prediction System — Academic Research Project
      </footer>

    </div>
  );
}
