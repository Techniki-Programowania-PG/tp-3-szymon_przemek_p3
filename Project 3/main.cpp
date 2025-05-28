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

class ResultVector3D {
public:
    matplot::vector_2d x;
    matplot::vector_2d y;
    matplot::vector_2d z;

    ResultVector3D() = default;
    ResultVector3D(const matplot::vector_2d& x_plot, const matplot::vector_2d& y_plot, const matplot::vector_2d& z_plot) : x(x_plot), y(y_plot), z(z_plot) {};

    ResultVector3D operator+(const ResultVector3D& other) const {
        using namespace matplot;

        int samples = x[0].size();

        if (samples != other.x[0].size()) {
            std::cout << "sizes differ! ";
            return ResultVector3D(x, y, z);
        }
        matplot::vector_2d Z(samples, std::vector<double>(samples));
        for (int a = 0; a < samples; ++a) {
            for (int b = 0; b < samples; ++b) {
                Z[a][b] = z[a][b] + other.z[a][b];
            }
        }

        return ResultVector3D(x, y, Z);
    }

    ResultVector3D operator*(double scalar) const {
        int samples = x[0].size();

        matplot::vector_2d Z(samples, std::vector<double>(samples));
        for (int a = 0; a < samples; ++a) {
            for (int b = 0; b < samples; ++b) {
                Z[a][b] = z[a][b] * scalar;
            }
        }

        return ResultVector3D(x, y, Z);
    }

    ResultVector3D mul(const ResultVector3D& other) const {
        int samples = x[0].size();

        if (samples != other.x[0].size()) {
            std::cout << "sizes differ! ";
            return ResultVector3D(x, y, z);
        }

        matplot::vector_2d Z(samples, std::vector<double>(samples));
        for (int a = 0; a < samples; ++a) {
            for (int b = 0; b < samples; ++b) {
                Z[a][b] = z[a][b] * other.z[a][b];
            }
        }

        return ResultVector3D(x, y, Z);
    }

};

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

    ResultVector3D mul(ResultVector plot2, double startX, double endX, double startY, double endY, int samples) {
        using namespace matplot;

        auto [X, Y] = meshgrid(linspace(startX, endX, samples), linspace(startY, endY, samples));
        matplot::vector_2d Z(samples, std::vector<double>(samples));
        for (int a = 0; a < samples; ++a) {
            for (int b = 0; b < samples; ++b) {
                Z[a][b] = y[b] * plot2.y[a];
            }
        }

        ResultVector3D result(X, Y, Z);
        return result;
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
            })
        .def("mul", &ResultVector::mul, py::arg("plot"), py::arg("startX"), py::arg("endX"), py::arg("startY"), py::arg("endY"), py::arg("samples"));


    py::class_<ResultVector3D>(m, "ResultVector3D")
        .def(py::init<>())
        .def(py::init<const matplot::vector_2d&, const matplot::vector_2d&, const matplot::vector_2d&>())
        .def_readwrite("x", &ResultVector3D::x)
        .def_readwrite("y", &ResultVector3D::y)
        .def_readwrite("z", &ResultVector3D::z)
        .def("__add__", &ResultVector3D::operator+)
        .def("__mul__", &ResultVector3D::operator*)
        .def("__rmul__", [](const ResultVector3D& vec, double scalar) { return vec * scalar; })
        .def("mul", &ResultVector3D::mul, py::arg("plot"));


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

    m.def("surf", [](ResultVector3D plot) {
        using namespace matplot;
        figure();
        surf(plot.x, plot.y, plot.z);
        xlabel("x");
        ylabel("y");
        zlabel("z");
        axis(on);
        grid(on);
        show();
        }, py::arg("plot3D"), R"pbdoc(
        Show surface/2D signal
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
        }, py::arg("plot"), py::arg("noise_level"), R"pbdoc(
            Aply gaussian noise to plot
    )pbdoc");

    m.def("noise2D", [](ResultVector3D plot, double power) {
        std::default_random_engine generator(std::random_device{}());
        std::normal_distribution<double> noise_dist(0.0, power * 0.02);

        for (int i = 0; i < plot.z.size(); ++i) {
            for (int j = 0; j < plot.z[i].size(); ++j) {
                double pure = plot.z[i][j];
                double noise = noise_dist(generator);
                plot.z[i][j] = (pure + noise);
            }
        }
        return plot;
        }, py::arg("plot"), py::arg("noise_level"), R"pbdoc(
            Apply gaussian noise to surface/2D signal
    )pbdoc");

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
        }, py::arg("plot"), R"pbdoc(
            Apply filter to plot
    )pbdoc");

    m.def("filter2Dexample", [](int samples) {
        using namespace matplot;
        std::default_random_engine generator(std::random_device{}());
        std::normal_distribution<double> noise_dist(0.0, 10.0);
        auto [X, Y] = meshgrid(linspace(-3, +3, samples), linspace(-3, +3, samples));
        auto testZ = transform(X, Y, [](double x, double y) {
            return (10 * 2 + pow(x, 2) - 10 * cos(2 * pi * x) + pow(y, 2) -
                10 * cos(2 * pi * y));
            });
        figure();
        surf(X, Y, testZ);
        show();
        auto Z = transform(X, Y, [generator, noise_dist](double x, double y) mutable {
            double noise = noise_dist(generator);
            return (10 * 2 + pow(x, 2) - 10 * cos(2 * pi * x) + pow(y, 2) -
                10 * cos(2 * pi * y)) + noise;
            });
        figure();
        surf(X, Y, Z);

        show();

        std::vector<std::vector<double>> gausss2dFilter = {
    {0.0039, 0.0156, 0.0234, 0.0156, 0.0039},
    {0.0156, 0.0625, 0.0938, 0.0625, 0.0156},
    {0.0234, 0.0938, 0.1406, 0.0938, 0.0234},
    {0.0156, 0.0625, 0.0938, 0.0625, 0.0156},
    {0.0039, 0.0156, 0.0234, 0.0156, 0.0039}
        };

        double output;
        auto Z_filtered = Z;
        int offset = gausss2dFilter.size() / 2;

        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < samples; ++j) {
                output = 0.0;
                for (int m = 0; m < gausss2dFilter.size(); ++m) {
                    for (int n = 0; n < gausss2dFilter[m].size(); ++n) {
                        int iPadding = i - offset + m;
                        if (iPadding < 0) iPadding = 0;
                        if (iPadding >= samples) iPadding = samples - 1;

                        int jPadding = j - offset + n;
                        if (jPadding < 0) jPadding = 0;
                        if (jPadding >= samples) jPadding = samples - 1;

                        output += Z[iPadding][jPadding] * gausss2dFilter[m][n];
                    }
                }
                Z_filtered[i][j] = output;
            }
        }
        figure();
        surf(X, Y, Z_filtered);
        show();

        }, py::arg("samples"), R"pbdoc(
            example of a 2D filter
            2D signal -> apply gaussian noise -> 2D filter
    )pbdoc");

    m.def("filter2D", [](ResultVector3D plot, int samples) {
        using namespace matplot;
        std::default_random_engine generator(std::random_device{}());
        std::normal_distribution<double> noise_dist(0.0, 10.0);

        std::vector<std::vector<double>> gausss2dFilter = {
    {0.0039, 0.0156, 0.0234, 0.0156, 0.0039},
    {0.0156, 0.0625, 0.0938, 0.0625, 0.0156},
    {0.0234, 0.0938, 0.1406, 0.0938, 0.0234},
    {0.0156, 0.0625, 0.0938, 0.0625, 0.0156},
    {0.0039, 0.0156, 0.0234, 0.0156, 0.0039}
        };

        double output;
        auto Z_filtered = plot.z;
        int offset = gausss2dFilter.size() / 2;

        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < samples; ++j) {
                output = 0.0;
                for (int m = 0; m < gausss2dFilter.size(); ++m) {
                    for (int n = 0; n < gausss2dFilter[m].size(); ++n) {
                        int iPadding = i - offset + m;
                        if (iPadding < 0) iPadding = 0;
                        if (iPadding >= samples) iPadding = samples - 1;

                        int jPadding = j - offset + n;
                        if (jPadding < 0) jPadding = 0;
                        if (jPadding >= samples) jPadding = samples - 1;

                        output += plot.z[iPadding][jPadding] * gausss2dFilter[m][n];
                    }
                }
                Z_filtered[i][j] = output;
            }
        }
        ResultVector3D result(plot.x, plot.y, Z_filtered);
        return result;

        }, py::arg("plot"), py::arg("samples"), R"pbdoc(
            Filter surface/2D signal with Gaussian filter -> return surface/2D signal
    )pbdoc");

    m.def("multiplyPlots2D", [](ResultVector plot1, ResultVector plot2, double startX, double endX, double startY, double endY, int samples) {
        using namespace matplot;

        auto [X, Y] = meshgrid(linspace(startX, endX, samples), linspace(startY, endY, samples));
        matplot::vector_2d Z(samples, std::vector<double>(samples));
        for (int a = 0; a < samples; ++a) {
            for (int b = 0; b < samples; ++b) {
                Z[a][b] = plot1.y[b] * plot2.y[a];
            }
        }

        ResultVector3D result(X, Y, Z);
        return result;
        }, py::arg("plot1"), py::arg("plot2"), py::arg("startX"), py::arg("endX"), py::arg("startY"), py::arg("endY"), py::arg("samples"), R"pbdoc(
            Multiply two plots -> returns surface/2D signal
    )pbdoc");

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
