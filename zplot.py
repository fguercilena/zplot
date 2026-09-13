from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import bisect
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


M1 = np.array(
    [
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ]
)
M1inv = np.linalg.inv(M1)

M2 = np.array(
    [
        [0.2104542553, +0.7936177850, -0.0040720468],
        [+1.9779984951, -2.4285922050, +0.4505937099],
        [+0.0259040371, +0.7827717662, -0.8086757660],
    ]
)
M2inv = np.linalg.inv(M2)

p_xyy = np.array(
    [
        [0.64, 0.33, 0.2126],
        [0.30, 0.60, 0.7152],
        [0.15, 0.06, 0.0722],
    ]
).T
invM = (
    np.array(
        [
            p_xyy[2] / p_xyy[1] * p_xyy[0],
            p_xyy[2],
            p_xyy[2] / p_xyy[1] * (1 - p_xyy[0] - p_xyy[1]),
        ]
    )
    * 100
)
# The above values are given only approximately, resulting in the fact that
# SRGB(1.0, 1.0, 1.0) is only approximately mapped into the reference
# whitepoint D65. Add a correction here.
whitepoints_cie1931_d65 = np.array([95.047, 100, 108.883])
correction = whitepoints_cie1931_d65 / np.sum(invM, axis=1)
invM = (invM.T * correction).T / 100


# A number of scalings f that map the magnitude [0, infty] to [0, 1] are
# possible. A desirable property is
#
# (1)  f(1 / r) = 1 - f(r).
#
# This makes sure that the representation of the inverse of a function is
# exactly as light as the original function is dark.
#
# The function
#
#     h(r) = r**a / (r**a + 1)
#
# fulfills (1). Disadvantage of this choice: h'(r)=0 at r=0 for all a > 1, so
# that h(r) has an inflection point in (0, 1) for all a > 1. For 0 < a < 1, the
# derivative at 0 is infinite. Only for a=1, the derivative is 1/2.
#
# For a=1.21268891, this function is very close to the popular alternative 2/pi
# * arctan(r), which also fulfills (1) and has derivative 1 / pi at r=0.
#
#  A general way of fulfilling (1) is
#
#  h(r) = phi(r) / (phi(r) + phi(1/r))
#
# for some phi(r). For phi(r)=sqrt(r), one gets h(r)= r / (r + 1). Ideas here
# are phi = log1p or log1p(log1p) (and so forth).
def default_scaling(r):
    return r**0.4 / (r**0.4 + 1)


def to_sRGB(
    F,
    scaling,
    saturation_adjustment=1.28,
):
    arg = np.angle(F)
    scaled_abs = scaling(np.abs(F))

    r0 = 0.08499547839164734
    r0 *= saturation_adjustment

    # Rotate the angles such a "green" color represents positive real values. The
    # rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for the
    # point 1.0.
    offset = 0.8936868 * np.pi
    # Map (r, angle) to a point in the color space; bicone mapping similar to what
    # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
    rd = r0 - r0 * 2 * np.abs(scaled_abs - 0.5)

    OkLab = np.array(
        [
            scaled_abs,
            rd * np.cos(arg + offset),
            rd * np.sin(arg + offset),
        ]
    )

    # From OkLab to XYZ times 100
    XYZ100 = np.tensordot(M1inv, np.tensordot(M2inv, OkLab, 1) ** 3, 1) * 100

    # From XYZ times 100 to sRGB
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    # https://www.color.org/srgb.pdf
    sRGB = (
        np.linalg.solve(invM, XYZ100.reshape(XYZ100.shape[0], -1)).reshape(XYZ100.shape)
        / 100
    ).clip(0.0, 1.0)

    # gamma correction
    a = 0.055
    ind = sRGB <= 0.0031308
    sRGB[ind] *= 12.92
    sRGB[~ind] = (1 + a) * sRGB[~ind] ** (1 / 2.4) - a

    return np.moveaxis(sRGB, 0, -1)


def add_colorbar_arg(cax):
    # Define the colormap
    RGB = to_sRGB(
        np.exp(1j * np.linspace(-np.pi, np.pi, 512)),
        lambda z: np.full_like(z, 0.5),
    )
    RGBA = np.pad(RGB, ((0, 0), (0, 1)), constant_values=1.0)
    newcmp = colors.ListedColormap(RGBA)

    # Define norm
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)

    # Setup the colorbar
    clb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp), cax=cax)

    # Aesthetics
    clb.set_label(r"${\rm Arg}$", rotation=0, ha="center", va="top")
    clb.ax.yaxis.set_label_coords(0.5, -0.03)
    clb.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
    clb.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "$0$", r"$+\pi/2$", r"$+\pi$"])


def add_colorbar_abs(cax, ticks, scaling, cmap=cm.gray):
    # Define norm
    norm = colors.Normalize(vmin=0, vmax=1)

    # Setup the colorbar
    clb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
    )

    # Aesthetics
    clb.set_label(r"${\rm Abs}$", rotation=0, ha="center", va="top")
    clb.ax.yaxis.set_label_coords(0.5, -0.03)

    ticks = np.asarray(list(ticks))
    scaled_ticks = scaling(ticks)
    clb.set_ticks([0.0, *scaled_ticks, 1.0])

    def float_to_latex(num):
        float_string = f"{num:.2g}"
        if "e" in float_string:
            base, exponent = float_string.split("e")
            base = float(base)
            exponent = int(exponent)
            if base != 1.0:
                return rf"${base:f} \times 10^{{{exponent:d}}}$"
            else:
                return rf"$10^{{{exponent:d}}}$"
        else:
            return "$" + float_string + "$"

    ticklabels = np.array([r"$0$"] + [float_to_latex(t) for t in ticks] + [r"$\infty$"])
    clb.set_ticklabels(ticklabels)


def zplot(
    f,
    x_range,
    y_range,
    mode="full",
    ax=None,
    scaling=default_scaling,
    verbose=False,
    levels=None,
):

    if mode in ("real", "imag", "reim"):
        # Set up grid
        if isinstance(x_range, tuple):
            assert x_range[0] < x_range[1]
            x_min, x_max, n_x = x_range
            xs = np.linspace(x_min, x_max, n_x + 1)
        else:
            xs = x_range

        if ax is None:
            ax = plt.gca()

        F = f(xs)

        if mode in ("real", "reim"):
            ax.plot(xs, np.real(F), linewidth="2", color="C0", label=r"Re")
        if mode in ("imag", "reim"):
            ax.plot(xs, np.imag(F), linewidth="2", color="C1", label=r"Im")

        ax.set_xlabel("z")
        ax.legend(loc="best")
        return

    # Get axes if none
    if ax is None:
        if mode != "3D":
            ax = plt.gca()
        else:
            gs = gridspec.GridSpec(1, 2, figure=plt.gcf(), width_ratios=[1, 0.05])
            ax = plt.gcf().add_subplot(gs[0, 0], projection="3d")
            cax = plt.gcf().add_subplot(gs[0, 1])
        ax.set_aspect("equal")
        colorbars = True
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
    else:
        if mode == "3D":
            if not isinstance(ax, Axes3D):
                print("3D mode requested, but axes are not 3D.")
                return
        colorbars = False

    # Set up grid
    if isinstance(x_range, tuple):
        assert x_range[0] < x_range[1]
        x_min, x_max, n_x = x_range
        xs = np.linspace(x_min, x_max, n_x + 1)
    else:
        xs = x_range

    if isinstance(y_range, tuple):
        assert y_range[0] < y_range[1]
        y_min, y_may, n_y = y_range
        ys = np.linspace(y_min, y_may, n_y + 1)
    else:
        ys = y_range

    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y

    # Evaluate function
    if verbose:
        F = np.empty_like(Z, dtype=complex)
        for j in range(len(ys)):
            F[:, j] = f(Z[:, j])
            print(f"Progress: {(j + 1) / len(ys) * 100:5.2f}%", end="\r")
        print(f"                 ", end="\r")
    else:
        F = f(Z)

    # Plot only absolute value
    if mode == "abs":

        absF = np.abs(F)
        scaled_absF = scaling(absF)

        im = ax.pcolormesh(
            X,
            Y,
            scaled_absF,
            cmap="viridis",
            shading="nearest",
            rasterized=True,
        )

        if levels is None:
            try:
                m, M = np.nanmin(scaled_absF), np.nanmax(scaled_absF)
                levels = np.geomspace(m * 1.1, M * 0.9, 7)
                for i, l in enumerate(levels):
                    levels[i] = bisect(
                        lambda x: scaling(x) - l, 0, 1e300, maxiter=10000
                    )
                levels = np.unique(levels)
            except Exception as e:
                levels = 7

        cs = ax.contour(X, Y, absF, levels=levels, colors="white", linewidths=0.75)
        ax.clabel(cs, fmt=lambda l: f"{l:.1e}", fontsize="small")

        if colorbars:
            ax_divider = make_axes_locatable(ax)
            cax1 = ax_divider.append_axes("right", size="5%", pad="4%")
            add_colorbar_abs(cax1, [], scaling, cmap="viridis")

        return

    # Get sRGB colors
    if mode in ("arg", "3D"):
        c_scaling = lambda r: np.full_like(r, 0.6)
        sRGB = to_sRGB(F, c_scaling, 1.98)
    else:
        sRGB = to_sRGB(F, scaling, 1.38)

    # Set nan values to white
    sRGB[np.any(np.isnan(sRGB), axis=-1)] = [1.0] * 3

    if mode in ("arg", "full"):
        ax.pcolormesh(
            X,
            Y,
            sRGB,
            shading="nearest",
            rasterized=True,
        )
    elif mode == "3D":
        ax.plot_surface(
            X, Y, scaling(np.abs(F)), facecolors=sRGB, rstride=1, cstride=1, shade=False
        )

    if colorbars:
        ax_divider = make_axes_locatable(ax)
        if mode == "full":
            cax1 = ax_divider.append_axes("right", size="5%", pad="4%")
            add_colorbar_abs(cax1, [], scaling)
        if mode in ("arg", "full"):
            cax2 = ax_divider.append_axes("right", size="5%", pad="4%")
            add_colorbar_arg(cax2)
        else:
            add_colorbar_arg(cax)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        prog="zplot.py",
        description="Plot a function on the complex plane in various ways, also with domain coloring.",
        epilog="Federico Maria Guercilena (2026)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "expr",
        type=str,
        help="Math expression with independent variable 'z' (e.g. 'sin(z)'). All functions from numpy are available. Use the special value 'test' to get a few example plots.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=+5,
        help="Extension of plot around the origin",
    )
    parser.add_argument(
        "-n",
        "--n_points",
        type=int,
        default=201,
        help="Number of points in [0, radius]",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="full",
        choices=["real", "imag", "reim", "full", "abs", "arg", "3D"],
        help="""real: 1D plot, real part |
        imag: 1D plot, imaginary part |
        reim: 1D plot, real and imaginary parts |
        full: complex plot, domain coloring |
        abs: complex plot of absolute value |
        arg: complex plot of argument |
        3D: complex 3D plot""",
    )
    parser.add_argument(
        "-l",
        "--levels",
        type=float,
        default=None,
        nargs="*",
        help="List of levels for absolute value contour plot. Only for mode=='abs'",
    )
    args = parser.parse_args()

    if args.expr != "test":
        from numpy import *
        from scipy.special import *

        f = lambda z: eval(args.expr)
        z_range = (-args.radius, +args.radius, args.n_points)
        if args.mode == "3D":
            z_range = (-args.radius, +args.radius, 51)
        zplot(f, z_range, z_range, mode=args.mode, verbose=True, levels=args.levels)
        plt.show()

    else:

        from scipy.special import gamma

        def moebius(z):
            return (z - (1 + 1 * 1j)) / (z - (-1 - 1 * 1j))

        zplot(lambda z: gamma(z + 1j), (-5, 5, 1501), None, mode="reim")
        plt.gcf().suptitle(
            r"gamma(z + i), 1D plot with automatically generated figure/axes"
        )
        plt.show()

        zplot(moebius, (-5, 5, 501), (-5, 5, 501))
        plt.gcf().suptitle(
            r"$\frac{z - (1 + i)}{z - (-1 - i)}$, full plot with automatically generated figure/axes"
        )
        plt.show()

        zplot(moebius, (-5, 5, 501), (-5, 5, 501), mode="abs")
        plt.gcf().suptitle(
            r"$\frac{z - (1 + i)}{z - (-1 - i)}$, abs plot with automatically generated figure/axes"
        )
        plt.show()

        zplot(moebius, (-5, 5, 501), (-5, 5, 501), mode="arg")
        plt.gcf().suptitle(
            r"$\frac{z - (1 + i)}{z - (-1 - i)}$, arg plot with automatically generated figure/axes"
        )
        plt.show()

        zplot(moebius, (-5, 5, 51), (-5, 5, 51), mode="3D")
        plt.gcf().suptitle(
            r"$\frac{z - (1 + i)}{z - (-1 - i)}$, arg plot with automatically generated figure/axes"
        )
        plt.show()

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 0.05, 0.05])
        ax = fig.add_subplot(gs[0, 0])
        cax1 = fig.add_subplot(gs[0, 1])
        cax2 = fig.add_subplot(gs[0, 2])
        zplot(gamma, (-5, 5, 401), (-5, 5, 401), ax=ax)
        add_colorbar_abs(cax1, [], default_scaling)
        add_colorbar_arg(cax2)
        fig.suptitle("Gamma function, full plot with external figure")
        plt.show()

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        cax1 = fig.add_subplot(gs[0, 1])
        zplot(gamma, (-5, 5, 51), (-5, 5, 51), mode="3D", ax=ax)
        add_colorbar_arg(cax1)
        fig.suptitle("Gamma function, 3D plot with external figure")
        plt.show()
