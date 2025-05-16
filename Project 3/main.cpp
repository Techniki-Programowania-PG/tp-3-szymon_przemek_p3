#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <matplot/matplot.h>
#include <vector>
#include <cmath>
#include <complex>

constexpr std::complex<double> i(0.0, 1.0);
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;
using namespace matplot;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Python library for signal processing based on example pybind11 plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
           sinus(f,A)
           cosinus(f,A)
    )pbdoc";

    m.def("add", &add,
        py::arg("i"), py::arg("j"),  R"pbdoc(
        Add two numbers
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
    )pbdoc");

    m.def("sin", [](double f, double y, int start, int end, int samples) {
        
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

        return ysin;    
    }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create sinus plot
    )pbdoc");

    m.def("cos", [](double f, double y, int start, int end, int samples) {
        
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

        return ycos;
        }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create cosinus plot
    )pbdoc");

    m.def("sqrwave", [](double f, double A, int start, int end, int sample) {
        
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

    m.def("sawwave", [](double f, double A, int start, int end, int sample) {
        
        std::vector<double> fsaw = linspace(start, end, sample);
        std::vector<double> ysaw;
        for (double val : fsaw) {
                ysaw.push_back(((std::fmod((f*(val/pi)),(2.0)))-1)*A) << endl;
                return 0;
        }
        plot(fsaw, ysaw);
        xlabel("x");
        ylabel("saw q8wave(x)");
        grid(on);
        show();

        return 0;
        }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create sawwave plot
    )pbdoc");

    m.def("fourier", [](std::vector<double> testPlot, int start, int end, int seqNr) {
        if (testPlot.size() != seqNr) {
            std::cout << "incorrect sample size, should be: " << testPlot.size();
        }
        std::vector<double> x = linspace(start, end, seqNr);
        std::vector<double> y;
        for (int k = 0; k < seqNr; ++k) {
            std::complex<double> fourierX = 0;
            for (int n = 0; n < seqNr; ++n) {
                double angle = -2.0 * pi * (double(k) / double(seqNr)) * n;
                fourierX += testPlot[n] * (std::exp(i * angle));
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

    m.def("inv_fourier", [](std::vector<double> testPlot, int start, int end, int seqNr) {
        if (testPlot.size() != seqNr) {
            std::cout << "incorrect sample size, should be: " << testPlot.size() << endl;
            return 0;
        }
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
        }, py::arg("DFT"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Inverted discrete Fourier transform
    )pbdoc");

//#ifdef VERSION_INFO
//    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
//#else
 //   m.attr("__version__") = "dev";
//#endif
    m.attr("__version__") = "dev";
}