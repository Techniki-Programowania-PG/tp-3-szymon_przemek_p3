#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <matplot/matplot.h>
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

constexpr std::complex<double> i(0.0, 1.0);
const std::vector<std::complex<double>> cplx0;
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

class ResultVector {
public:
    std::vector<double> x;
    std::vector<double> y;
    std::vector<std::complex<double>> j;

    ResultVector() = default;
    ResultVector(const std::vector<double>& x_plot, const std::vector<double>& y_plot, const std::vector<std::complex<double>>& complex) :x(x_plot), y(y_plot), j(complex) {};

    ResultVector operator+(const ResultVector& other) const { //jeœli ta sama wielkoœæ wektorów to zsumuj je za pomoc¹ (przeci¹¿onego) "+"
        std::vector<double> sum(y.size());
        if (x != other.x) {
            std::cout << "sizes differ! ";
            sum = { 0 };
            return ResultVector(x, sum, cplx0);
        }
        std::transform(y.begin(), y.end(), other.y.begin(), sum.begin(),
            [](double v1, double v2) {return (v1 + v2); });
        return ResultVector(x, sum, cplx0);
    }

    ResultVector operator*(double scalar) const {//this -> Result vector, czyli x i y, other -> scalar
        std::vector<double>multiplication(y.size());
        std::transform(y.begin(), y.end(), multiplication.begin(),
            [scalar](double v0) {return (scalar * v0); });
        return ResultVector(x, multiplication, cplx0); // wynik wektor * skalar
    }
};

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    py::class_<ResultVector>(m, "ResultVector")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&, const std::vector<std::complex<double>>&>())
        .def_readwrite("x", &ResultVector::x)
        .def_readwrite("y", &ResultVector::y)
        .def_readwrite("j", &ResultVector::j)
        .def("__add__", &ResultVector::operator+) //vector + vector, jeœli te same end start i samples
        .def("__mul__", &ResultVector::operator*) //vector * scalar
        .def("__rmul__", [](const ResultVector& vec, double scalar) { //rmul bierze na odwrót argumenty, czyli taki zapis odpowiada scalar * vector
        return vec * scalar;
            });


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
           Gaussian noise
           1D & 2D filter
           correlation between two signals
    )pbdoc";

    m.def("plot", [](ResultVector plot) {
        using namespace matplot;
        figure();
        matplot::plot(plot.x, plot.y);
        xlabel("x");
        ylabel("y(x)");
        axis(on);
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
        
        std::vector<double> fsin = matplot::linspace(start, end, samples);
        std::vector<double> ysin;

        for (double val : fsin) {
            ysin.push_back(std::sin(val*f)*y);
        }
        ResultVector sinPlot(fsin, ysin, cplx0);
        return sinPlot;
    }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create sinus plot
    )pbdoc");

    m.def("cos", [](double f, double y, double start, double end, int samples) {
        
        std::vector<double> fcos = matplot::linspace(start, end, samples);
        std::vector<double> ycos;
        for (double val : fcos) {
            ycos.push_back(std::cos(val*f) * y);
        }
        ResultVector cosPlot(fcos, ycos, cplx0);
        return cosPlot;
        }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create cosinus plot
    )pbdoc");

    m.def("sqrwave", [](double f, double A, double start, double end, int sample) {
        
        std::vector<double> fsqw = matplot::linspace(start, end, sample);
        std::vector<double> ysqw;
        for (double val : fsqw) {
            if (std::sin(val * f) < 0) ysqw.push_back(-A);
            else if (std::sin(val * f) > 0) ysqw.push_back(A);
            else ysqw.push_back(0);
        }
        ResultVector sqwPlot(fsqw, ysqw, cplx0);
        return sqwPlot;
        }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create square wave plot
    )pbdoc");

    m.def("sawwave", [](double f, double A, double start, double end, int sample) {        
        std::vector<double> fsaw = matplot::linspace(start, end, sample);
        std::vector<double> ysaw;
        for (double val : fsaw) {
            ysaw.push_back(((std::fmod((f * (val / matplot::pi)), (2.0))) - 1) * A);
        }
        ResultVector sawPlot(fsaw, ysaw, cplx0);
        return sawPlot;
        }, py::arg("frequency"), py::arg("amplitude"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Create sawwave plot
    )pbdoc");

    m.def("fourier", [](ResultVector testPlot, double start, double end, int seqNr) {
        std::vector<double> x = matplot::linspace(start, end, seqNr);
        std::vector<double> y;
        std::vector<std::complex<double>> complexVec;
        for (int k = 0; k < seqNr; ++k) {
            std::complex<double> fourierX = 0;
            for (int n = 0; n < seqNr; ++n) {
                double angle = 2.0 * matplot::pi * (double(k) / double(seqNr)) * n;
                fourierX += testPlot.y[n] * (std::exp(-i * angle));
            }
            complexVec.push_back(fourierX);
            y.push_back(std::abs(fourierX));
        }
        ResultVector dftPlot(x, y, complexVec);
        return dftPlot;
        }, py::arg("plot"), py::arg("start"), py::arg("end"), py::arg("samples"), R"pbdoc(
        Discrete Fourier transform
    )pbdoc");

    m.def("frequenciesDFT", [](ResultVector testPlot, int end, int sampleSize, int range2pi) {
        if (range2pi * end + 1 > sampleSize - 1 || range2pi * end < 1)
        {
            range2pi = 1;
            end = sampleSize - 1;
        }
        std::vector<double> x = matplot::linspace(0, end, range2pi*end+1);
        std::vector<double> y;

        for (int i = 0; i < range2pi * (end + 1); i++)
            y.push_back(testPlot.y[i]);

        ResultVector shortPlot(x, y, testPlot.j);
        return shortPlot;

        }, py::arg("plot"), py::arg("end"), py::arg("sampleSize"), py::arg("rangeBy2pi"), R"pbdoc(
        transform DFT plot to see the frequencies clearer (works best for sine & cosine functions)

        rangeBy2pi is the original range of the signal, before the DFT transformation, divided by 2pi

        frequencies only show if they are equal to (n/rangeBy2pi), where n is any natural number
    )pbdoc");

    m.def("inv_fourier", [](ResultVector testPlot, double start, double end) {
        int seqNr = testPlot.x.size();
        std::vector<double> x = matplot::linspace(start, end, seqNr);
        std::vector<double> y;
        std::vector<std::complex<double>> complexVec;
        for (int k = 0; k < seqNr; ++k) {
            std::complex<double> inv_fourierX = 0;
            for (int n = 0; n < seqNr; ++n) {
                double angle = 2.0 * matplot::pi * (double(k) / double(seqNr)) * n;
                inv_fourierX += testPlot.j[n] * (std::exp(i * angle));
            }
            y.push_back((1.0 / seqNr) * (inv_fourierX.real()));
            complexVec.push_back((1.0 / seqNr) * inv_fourierX);
        }
        ResultVector idftPlot(x, y, complexVec);
        return idftPlot;
        }, py::arg("DFT"), py::arg("start"), py::arg("end"), R"pbdoc(
        Inverted discrete Fourier transform
    )pbdoc");

    m.def("noise", [](ResultVector plot, double power) {
        std::default_random_engine generator(std::random_device{}());
        std::normal_distribution<double> noise_dist(0.0, power * 0.02);

        for (int i = 1; i < plot.y.size(); ++i) {
            double pure = plot.y[i];
            double noise = noise_dist(generator);
            plot.y[i] = (pure + noise);
        }
        return plot;
        }, py::arg("plot"), py::arg("noise_level"));

    m.def("filter1D", [](ResultVector testPlot) {
            std::vector<double> filter = {0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006};// filtr gaussa
            std::vector<double> out_plot;
            int filter_size = filter.size();
            int offset = filter_size / 2;
            int n = testPlot.y.size();
            double sum;
            for (int k = 0; k < n; ++k) {
                sum = 0;
                for (int r = 0; r < filter_size; ++r) {
                    int plot_index = k + r - offset;
                    if (plot_index >= 0 && plot_index < n) {
                        sum += (testPlot.y[plot_index] * filter[r]);
                    } else {
                        int padding;
                        if (plot_index < 0) padding = 0;
                        else if (plot_index > (n - 1)) padding = (n - 1);
                        else padding = plot_index;
                        sum += testPlot.y[padding] * filter[r];
                    }
                }
                out_plot.push_back(sum);
            }
            ResultVector filteredPlot(testPlot.x, out_plot, testPlot.j);
            return filteredPlot;
        }, py::arg("plot"));

    m.def("filter2D", []() {
        using namespace matplot;
        auto [X, Y] = meshgrid(linspace(-3, +3, 50), linspace(-3, +3, 50));
        auto Z = transform(X, Y, [](double x, double y) {
            return 10 * 2 + pow(x, 2) - 10 * cos(2 * pi * x) + pow(y, 2) -
                10 * cos(2 * pi * y);
            });
        surf(X, Y, Z);

        show();
        return Z;
        });

    m.def("correlation", [](ResultVector testPlot1, ResultVector testPlot2) {
        std::vector<double> out_plot;
        if(testPlot1.x != testPlot2.x) {
            std::cout << "sizes differ!";
            out_plot = { 0 };
            return ResultVector(testPlot1.x, out_plot, cplx0);
	}
	int plotSize = testPlot1.y.size();
        long double sum;
        for (int n = 0; n < plotSize; ++n) {
            sum = 0;
            for (int k = 0; k < plotSize; ++k) {
                if (k + n < plotSize)
		    sum += (testPlot1.y[k] * testPlot2.y[k + n]);
            }
	    out_plot.push_back(sum);
        }

	ResultVector correlationPlot(testPlot1.x, out_plot, cplx0);
        return correlationPlot;
    	}, py::arg("plot1"), py::arg("plot2"), R"pbdoc(
            Create correlation plot
    )pbdoc");

    m.attr("__version__") = "dev";
}
