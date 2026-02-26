import React from "react";

const Explainability = () => {
  return (
    <div className="min-h-screen bg-gray-50 px-10 py-10">

      {/* ================= HEADER ================= */}
      <div className="mb-10 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">
            Model Explainability
          </h1>
          <p className="text-gray-600 mt-2">
            Understand how the XGBoost model makes predictions using SHAP
            (SHapley Additive Explanations).
          </p>
        </div>

        <div className="bg-green-100 text-green-700 px-4 py-1 rounded-full text-sm font-medium">
          Model: XGBoost | Explainability: SHAP
        </div>
      </div>

      {/* ================= GLOBAL EXPLAINABILITY ================= */}
      <div className="mb-14">
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">
          Global Explainability
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

          {/* Feature Importance */}
          <div className="bg-white rounded-2xl shadow-md p-6 hover:shadow-lg transition duration-300">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Mean SHAP Feature Importance
            </h3>

            {/* Replace with backend image later */}
            <div className="bg-gray-100 rounded-xl h-72 flex items-center justify-center text-gray-400">
              Global SHAP Bar Plot
            </div>

            <p className="text-sm text-gray-600 mt-4 leading-relaxed">
              Shows the overall contribution of each feature across all disease
              classes. Higher mean SHAP values indicate stronger influence on
              model predictions.
            </p>
          </div>

          {/* Summary Plot */}
          <div className="bg-white rounded-2xl shadow-md p-6 hover:shadow-lg transition duration-300">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              SHAP Summary (Beeswarm) Plot
            </h3>

            {/* Replace with backend image later */}
            <div className="bg-gray-100 rounded-xl h-72 flex items-center justify-center text-gray-400">
              SHAP Summary Plot
            </div>

            <p className="text-sm text-gray-600 mt-4 leading-relaxed">
              Displays how individual feature values affect the model output.
              Red indicates higher feature values and blue indicates lower
              feature values.
            </p>
          </div>

        </div>
      </div>

      {/* ================= LOCAL EXPLAINABILITY ================= */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">
          Local Explainability
        </h2>

        <div className="bg-white rounded-2xl shadow-md p-8 hover:shadow-lg transition duration-300">

          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            SHAP Force Plot (Individual Prediction)
          </h3>

          {/* Replace with force plot from backend later */}
          <div className="bg-gray-100 rounded-xl h-80 flex items-center justify-center text-gray-400">
            SHAP Force Plot Visualization
          </div>

          <p className="text-sm text-gray-600 mt-4 leading-relaxed">
            This visualization explains how each feature contributes positively
            or negatively to a specific prediction. Features pushing the
            prediction higher appear on one side, while those reducing it appear
            on the opposite side.
          </p>
        </div>
      </div>

    </div>
  );
};

export default Explainability;