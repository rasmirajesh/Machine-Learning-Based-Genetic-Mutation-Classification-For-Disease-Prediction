import React, { useEffect, useState } from "react";

const Explainability = () => {
  const [globalPlots, setGlobalPlots] = useState(null);
  const [localPlot, setLocalPlot] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(true);

  // ================= GLOBAL SHAP =================
  useEffect(() => {
    fetch("http://127.0.0.1:8000/explain")
      .then((res) => res.json())
      .then((data) => {
        setGlobalPlots(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading global SHAP:", err);
        setLoading(false);
      });
  }, []);

  // ================= LOCAL SHAP =================
  const handleLocalExplain = async () => {
    if (!file) {
      alert("Please upload a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/explain-local",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();
      setLocalPlot(data.force_plot);
    } catch (error) {
      console.error("Error generating local SHAP:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 px-10 py-10">
      {/* ================= HEADER ================= */}
      <div className="mb-10 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">
            Model Explainability
          </h1>
          <p className="text-gray-600 mt-2">
            Understand how the XGBoost model makes predictions using SHAP.
          </p>
        </div>

        <div className="bg-green-100 text-green-700 px-4 py-1 rounded-full text-sm font-medium">
          Model: XGBoost | SHAP
        </div>
      </div>

      {/* ================= GLOBAL ================= */}
      {loading ? (
        <div className="text-center text-gray-500">
          Loading SHAP visualizations...
        </div>
      ) : globalPlots ? (
        <>
          <h2 className="text-2xl font-semibold text-gray-800 mb-6">
            Global Explainability
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-14">
            {/* Feature Importance */}
            <div className="bg-white rounded-2xl shadow-md p-6">
              <h3 className="text-lg font-semibold mb-4">
                Mean SHAP Feature Importance
              </h3>

              <img
                src={`data:image/png;base64,${globalPlots.global_plot}`}
                alt="Global SHAP"
                className="rounded-xl w-full"
              />
            </div>

            {/* Summary Plot */}
            <div className="bg-white rounded-2xl shadow-md p-6">
              <h3 className="text-lg font-semibold mb-4">
                SHAP Summary Plot
              </h3>

              <img
                src={`data:image/png;base64,${globalPlots.summary_plot}`}
                alt="Summary SHAP"
                className="rounded-xl w-full"
              />
            </div>
          </div>
        </>
      ) : (
        <div className="text-red-500">
          Failed to load global SHAP plots.
        </div>
      )}

      {/* ================= LOCAL ================= */}
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">
        Local Explainability
      </h2>

      <div className="bg-white rounded-2xl shadow-md p-8">
        <h3 className="text-lg font-semibold mb-4">
          SHAP Force Plot (Individual Prediction)
        </h3>

        {/* Upload */}
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
          className="mb-4"
        />

        <button
          onClick={handleLocalExplain}
          className="bg-green-600 text-white px-4 py-2 rounded-lg mb-6"
        >
          Generate Local Explanation
        </button>

        {/* Plot */}
        {localPlot ? (
          <img
            src={`data:image/png;base64,${localPlot}`}
            alt="Local SHAP"
            className="rounded-xl w-full"
          />
        ) : (
          <div className="bg-gray-100 rounded-xl h-64 flex items-center justify-center text-gray-400">
            Upload CSV and click Generate
          </div>
        )}

        <p className="text-sm text-gray-600 mt-4">
          This plot explains how each feature contributes to a specific
          prediction.
        </p>
      </div>
    </div>
  );
};

export default Explainability;