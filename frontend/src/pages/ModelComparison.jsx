import React from "react";

const ModelComparison = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-10">

      <h1 className="text-3xl font-bold text-gray-800 mb-8">
        Model Performance Comparison
      </h1>

      {/* Comparison Cards */}
      <div className="grid md:grid-cols-2 gap-8">

        {/* Random Forest */}
        <div className="bg-white rounded-2xl shadow-md p-6">
          <h2 className="text-xl font-semibold text-teal-700 mb-4">
            Random Forest
          </h2>

          <ul className="text-gray-600 space-y-2">
            <li>• Baseline classification model</li>
            <li>• Good overall accuracy</li>
            <li>• Risk Score identified as most influential feature</li>
            <li>• Moderate generalization</li>
          </ul>

          <div className="h-48 bg-gray-100 rounded-lg mt-6 flex items-center justify-center text-gray-400">
            Random Forest Confusion Matrix
          </div>
        </div>

        {/* XGBoost */}
        <div className="bg-white rounded-2xl shadow-md p-6">
          <h2 className="text-xl font-semibold text-teal-700 mb-4">
            XGBoost (Selected Model)
          </h2>

          <ul className="text-gray-600 space-y-2">
            <li>• Higher accuracy (~91%)</li>
            <li>• Better generalization</li>
            <li>• Strong feature importance ranking</li>
            <li>• Selected as final prediction model</li>
          </ul>

          <div className="h-48 bg-gray-100 rounded-lg mt-6 flex items-center justify-center text-gray-400">
            XGBoost Confusion Matrix
          </div>
        </div>

      </div>

      {/* Conclusion Section */}
      <div className="bg-white rounded-2xl shadow-md p-6 mt-10">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Final Model Selection
        </h2>
        <p className="text-gray-600 leading-relaxed">
          Based on comparative evaluation, XGBoost demonstrated better
          classification accuracy and generalization performance across
          disease classes. Therefore, it was selected as the final model
          for deployment and SHAP-based explainability.
        </p>
      </div>

    </div>
  );
};

export default ModelComparison;