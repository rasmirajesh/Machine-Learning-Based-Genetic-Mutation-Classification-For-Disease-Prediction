import { Link, useLocation } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();

  const navItem = (path, label) => (
    <Link
      to={path}
      className={`px-4 py-2 rounded-lg font-medium transition ${
        location.pathname === path
          ? "bg-green-600 text-white"
          : "text-gray-700 hover:bg-green-100"
      }`}
    >
      {label}
    </Link>
  );

  return (
    <nav className="bg-white shadow-sm px-10 py-4 flex justify-between items-center">
      <h1 className="text-xl font-bold text-green-700">
        GeneMutate AI
      </h1>

      <div className="flex gap-4">
        {navItem("/", "Home")}
        {navItem("/predict", "Predict")}
        {navItem("/explainability", "Explainability")}
        {navItem("/model-comparison", "Model Comparison")}
        {navItem("/about", "About")}
      </div>
    </nav>
  );
};

export default Navbar;