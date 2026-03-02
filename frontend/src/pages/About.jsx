import React from "react";

const About = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-10">
      <h1 className="text-3xl font-bold text-gray-800 mb-4">
        About GeneMutate AI
      </h1>
      <p className="text-gray-600">
        This project focuses on genetic mutation detection and disease prediction
        using Machine Learning models such as Random Forest and XGBoost,
        along with SHAP for explainability.
      </p>
    </div>
  );
};

export default About;