import { Link } from "react-router-dom";

export default function About() {
  return (
    <div className="min-h-screen bg-[#f5f7f9]">

      {/* HERO SECTION (same style as home) */}
      <div className="bg-[#0f8f6a] text-white py-24 px-20 text-center">
        <h1 className="text-5xl font-bold mb-6">
          About GeneMutate AI
        </h1>

        <p className="text-lg max-w-3xl mx-auto opacity-90">
          An AI-powered platform for genetic mutation detection and disease prediction
          using machine learning and explainable artificial intelligence.
        </p>
      </div>

      {/* CONTENT SECTION */}
      <div className="px-20 py-16">

        <div className="max-w-5xl text-gray-600 leading-relaxed space-y-6">

          <p>
            GeneMutate AI is an AI-powered platform designed to analyze genetic mutation
            data and predict potential hereditary diseases using machine learning.
            The system allows users to upload mutation datasets derived from genomic
            data and automatically analyzes them to identify patterns associated with
            specific genetic disorders.
          </p>

          <p>
            This project focuses on predicting the risk of hereditary diseases such as
            <strong> Sickle Cell Disease, Cystic Fibrosis, and Huntington’s Disease </strong>
            by leveraging powerful machine learning models including
            <strong> Random Forest </strong> and <strong> XGBoost</strong>.
            These models are trained on mutation-related features to classify and
            predict possible disease outcomes from genomic mutation data.
          </p>

          <p>
            To ensure transparency and interpretability, the system integrates
            <strong> SHAP (Shapley Additive Explanations)</strong> which helps explain how
            different genetic features influence the model's predictions. This enables
            users to understand the reasoning behind predictions rather than treating
            the model as a black-box system.
          </p>

          <p>
            The platform combines modern web technologies with artificial intelligence
            to create an interactive environment for genomic data analysis. The
            frontend is built using <strong>React and Tailwind CSS</strong>, while the
            backend uses <strong>FastAPI</strong> to process data and run machine
            learning models efficiently.
          </p>

          <p>
            This system is developed as an academic research project that demonstrates
            how machine learning and explainable AI techniques can be applied to
            genomic mutation analysis to support disease prediction and biomedical
            research.
          </p>

        </div>
      </div>

      {/* FOOTER */}
      <footer className="text-center text-sm text-gray-500 py-6 border-t">
        © 2026 Genetic Mutation Detection & Disease Prediction System — Academic Research Project
      </footer>

    </div>
  );
}