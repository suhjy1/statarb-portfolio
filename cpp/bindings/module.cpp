#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "statarb/backtest.hpp"
#include "statarb/ou.hpp"

namespace py = pybind11;

namespace {

template <typename T>
using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

statarb::ConstMatrix view2d(const Array<double>& a, const char* name) {
  if (a.ndim() != 2) {
    throw py::value_error(std::string(name) + " must be 2-D");
  }
  return {a.data(), static_cast<std::size_t>(a.shape(0)), static_cast<std::size_t>(a.shape(1))};
}

template <typename T>
std::span<const T> view1d(const Array<T>& a, std::size_t expected, const char* name) {
  if (a.ndim() != 1 || static_cast<std::size_t>(a.size()) != expected) {
    throw py::value_error(std::string(name) + " must be 1-D of length " + std::to_string(expected));
  }
  return {a.data(), expected};
}

py::array_t<double> to_numpy(const std::vector<double>& v) {
  return py::array_t<double>(static_cast<py::ssize_t>(v.size()), v.data());
}

py::array_t<double> to_numpy(const statarb::Matrix& m) {
  return py::array_t<double>({static_cast<py::ssize_t>(m.rows), static_cast<py::ssize_t>(m.cols)},
                             m.data.data());
}

}

PYBIND11_MODULE(statarb_cpp, m) {
  m.doc() = "statarb C++ kernels: batch OU estimation and portfolio simulation";

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const statarb::NotImplemented& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    }
  });

  m.def(
      "fit_ou_batch",
      [](const Array<double>& paths, double dt) {
        const auto view = view2d(paths, "paths");
        std::vector<statarb::OUFit> fits;
        {
          py::gil_scoped_release release;
          fits = statarb::fit_ou_batch(view, dt);
        }
        if (fits.size() != view.cols) {
          throw std::runtime_error("fit_ou_batch returned the wrong number of fits");
        }
        const auto n = static_cast<py::ssize_t>(fits.size());
        py::array_t<double> kappa(n), mean(n), sigma(n);
        py::array_t<bool> valid(n);
        for (py::ssize_t i = 0; i < n; ++i) {
          const auto& f = fits[static_cast<std::size_t>(i)];
          kappa.mutable_at(i) = f.kappa;
          mean.mutable_at(i) = f.m;
          sigma.mutable_at(i) = f.sigma;
          valid.mutable_at(i) = f.valid;
        }
        py::dict out;
        out["kappa"] = kappa;
        out["m"] = mean;
        out["sigma"] = sigma;
        out["valid"] = valid;
        return out;
      },
      py::arg("paths"),
      py::arg("dt"),
      "OU fit per column of paths [T x N]; dict of kappa, m, sigma, valid arrays.");

  py::class_<statarb::CostParams>(m, "CostParams")
      .def(py::init<>())
      .def_readwrite("stock_half_spread_bps", &statarb::CostParams::stock_half_spread_bps)
      .def_readwrite("etf_half_spread_bps", &statarb::CostParams::etf_half_spread_bps)
      .def_readwrite("impact_coef", &statarb::CostParams::impact_coef)
      .def_readwrite("borrow_bps_annual", &statarb::CostParams::borrow_bps_annual)
      .def_readwrite("capital", &statarb::CostParams::capital)
      .def_readwrite("liquidity_spreads", &statarb::CostParams::liquidity_spreads)
      .def_readwrite("spread_ref_dollar_volume", &statarb::CostParams::spread_ref_dollar_volume)
      .def_readwrite("spread_elasticity", &statarb::CostParams::spread_elasticity)
      .def_readwrite("spread_min_bps", &statarb::CostParams::spread_min_bps)
      .def_readwrite("spread_max_bps", &statarb::CostParams::spread_max_bps);

  py::class_<statarb::BacktestParams>(m, "BacktestParams")
      .def(py::init<>())
      .def_readwrite("execution_lag", &statarb::BacktestParams::execution_lag)
      .def_readwrite("min_trade", &statarb::BacktestParams::min_trade);

  m.def(
      "run_backtest",
      [](const Array<double>& targets,
         const Array<std::int64_t>& target_rows,
         const Array<double>& returns,
         const Array<double>& dollar_volume,
         const Array<std::uint8_t>& is_etf,
         const statarb::CostParams& costs,
         const statarb::BacktestParams& params) {
        const auto t = view2d(targets, "targets");
        const auto r = view2d(returns, "returns");
        const auto dv = view2d(dollar_volume, "dollar_volume");
        if (t.cols != r.cols || dv.rows != r.rows || dv.cols != r.cols) {
          throw py::value_error("targets, returns and dollar_volume must share instruments");
        }
        const auto rows = view1d(target_rows, t.rows, "target_rows");
        const auto etf = view1d(is_etf, r.cols, "is_etf");
        statarb::BacktestOutput out;
        {
          py::gil_scoped_release release;
          out = statarb::run_backtest(t, rows, r, dv, etf, costs, params);
        }
        py::dict d;
        d["weights"] = to_numpy(out.weights);
        d["gross_pnl"] = to_numpy(out.gross_pnl);
        d["costs"] = to_numpy(out.costs);
        d["net_pnl"] = to_numpy(out.net_pnl);
        d["turnover"] = to_numpy(out.turnover);
        return d;
      },
      py::arg("targets"),
      py::arg("target_rows"),
      py::arg("returns"),
      py::arg("dollar_volume"),
      py::arg("is_etf"),
      py::arg("costs"),
      py::arg("params"),
      "Backtest engine (see cpp/include/statarb/backtest.hpp); dict of NumPy arrays.");
}
