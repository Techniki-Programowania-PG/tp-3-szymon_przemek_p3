#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <matplot/matplot.h>
#include <vector>
#include <cmath>
#include <complex>

constexpr std::complex<double> i(0.0, 1.0);
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

class ResultVector {
public:
    std::vector<double> x;
    std::vector<double> y;

    ResultVector() = default;
    ResultVector(const std::vector<double>& x_plot, const std::vector<double>& y_plot) :x(x_plot), y(y_plot) {};

    ResultVector operator+(const ResultVector& other) const {
        std::vector<double> sum(y.size());
        if (x != other.x) {
            std::cout << "sizes differs! ";
            sum = { 0 };
            return ResultVector(x, sum);
        }
        std::transform(y.begin(), y.end(), other.y.begin(), sum.begin(),
            [](double v1, double v2) {return (v1 + v2); });
        return ResultVector(x, sum);
    }
};

int add(int i, int j) {
    return i + j;
}

std::vector<double> operator*(const std::vector<double>& vec, double scalar) {
    std::vector<double> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(),
        [scalar](double v) { return v * scalar; });
    return result;
}


std::vector<double> operator*(double scalar, const std::vector<double>& vec) {
    return vec * scalar;
}

//przenoszenie overloadingu do klasy


//std::vector<double> operator+(const std::vector<double>& vec1, const std::vector<double>& vec2) {
//    std::vector<double> sum(vec1.size());
//    if (vec1.size() != vec2.size()) {
//        std::cout << "sizes differs: " << vec1.size() << " != " << vec2.size();
//        sum = { 0 };
//        return sum;
//    }
//    std::transform(vec1.begin(), vec1.end(), vec2.begin(), sum.begin(),
//        [](double v1, double v2) {return (v1 + v2); });
//    return sum;
//}

namespace py = pybind11;
using namespace matplot;

PYBIND11_MODULE(_core, m) {
    py::class_<ResultVector>(m, "ResultVector")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&>())
        .def_readwrite("x", &ResultVector::x)
        .def_readwrite("y", &ResultVector::y)
        .def("__add__", &ResultVector::operator+);


    m.doc() = R"pbdoc(
        Python library for signal processing based on example pybind11 plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
           sinus
           cosinus
           square wave
           saw wave
           DFT - Discrete Fourier Transformation (and inversion)
    )pbdoc";

    m.def("plot", [](ResultVector plot) {
        matplot::plot(plot.x, plot.y);
        xlabel("x");
        ylabel("y(x)");
        grid(on);
        show();
        }, py::arg("plot"), R"pbdoc(
        Create function plot
    )pbdoc");

    m.def("add", &add,
        py::arg("i"), py::arg("j"),  R"pbdoc(
        Add two numbers
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
    )pbdoc");

    m.def("sin", [](double f, double y, double start, double end, int samples) {
        
        std::vector<double> fsin = linspace(start, end, samples);
        std::vector<double> ysin;
        for (double val : fsin) {
            ysin.push_back(std::sin(val*f)*y);
        }
        plot(fsin, ysin);
        xlabel("x");
        ylabel("sin(x)");
        grid(on);
        show();
        ResultVector sinPlot(fsin, ysin);
        return sinPlot;
    }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create sinus plot
    )pbdoc");

    m.def("cos", [](double f, double y, double start, double end, int samples) {
        
        std::vector<double> fcos = linspace(start, (end), samples);
        std::vector<double> ycos;
        for (double val : fcos) {
            ycos.push_back(std::cos(val*f) * y);
        }
        plot(fcos, ycos);
        xlabel("x");
        ylabel("cos(x)");
        grid(on);
        show();
        ResultVector cosPlot(fcos, ycos);
        return cosPlot;
        }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create cosinus plot
    )pbdoc");

    m.def("sqrwave", [](double f, double A, double start, double end, int sample) {
        
        std::vector<double> fsqw = linspace(start, end, sample);
        std::vector<double> ysqw;
        for (double val : fsqw) {
            if (std::sin(val * f) < 0) ysqw.push_back(-A);
            else if (std::sin(val * f) > 0) ysqw.push_back(A);
            else ysqw.push_back(0);
        }
        plot(fsqw, ysqw);
        xlabel("x");
        ylabel("sqwave(x)");
        grid(on);
        show();

        return 0;
        }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create square wave plot
    )pbdoc");

    m.def("sawwave", [](double f, double A, double start, double end, int sample) {        
        std::vector<double> fsaw = linspace(start, end, sample);
        std::vector<double> ysaw;
        for (double val : fsaw) {
            ysaw.push_back(((std::fmod((f * (val / pi)), (2.0))) - 1) * A);
                return 0;
        }
        plot(fsaw, ysaw);
        xlabel("x");
        ylabel("saw wave(x)");
        grid(on);
        show();

        return 0;
        }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create sawwave plot
    )pbdoc");

    m.def("fourier", [](ResultVector testPlot, double start, double end, int seqNr) {
        std::vector<double> x = linspace(start, end, seqNr);
        std::vector<double> y;
        for (int k = 0; k < seqNr; ++k) {
            std::complex<double> fourierX = 0;
            for (int n = 0; n < seqNr; ++n) {
                double angle = -2.0 * pi * (double(k) / double(seqNr)) * n;
                fourierX += testPlot.y[n] * (std::exp(i * angle));
            }
            y.push_back(std::abs(fourierX));
        }
        plot(x, y);
        xlabel("t");
        ylabel("fft(t)");
        grid(on);
        show();

        return y;
        }, py::arg("plot"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Discrete Fourier transform
    )pbdoc");

    m.def("inv_fourier", [](std::vector<double> testPlot, double start, double end) {
        int seqNr = testPlot.size();
        std::vector<double> x = linspace(start, end, seqNr);
        std::vector<double> y;
        for (int k = 0; k < seqNr; ++k) {
            std::complex<double> inv_fourierX = 0;
            for (int n = 0; n < seqNr; ++n) {
                double angle = 2.0 * pi * (double(k) / double(seqNr)) * n;
                inv_fourierX += testPlot[n] * (std::exp(i * angle));
            }
            y.push_back((1.0/seqNr)*inv_fourierX.real());
        }
        plot(x, y);
        xlabel("t");
        ylabel("inv_fft(t)");
        grid(on);
        show();

        return 0;
        }, py::arg("DFT"), py::arg("start"), py::arg("end"), R"pbdoc(
        Inverted discrete Fourier transform
    )pbdoc");

    m.attr("__version__") = "dev";
}