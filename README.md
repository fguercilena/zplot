# zplot
Python scrpit to quickly plot a function on the complex plane in various ways, also with domain coloring.

## Requirements
Python 3
numpy
matplotlib
scipy

## Functionality

For starters, simply run `python zplot.py "test"` to get an idea of what the script can do.


### Importing *zplot.py*

When imported, *zplot.py* exposes the function:
```
zplot(
    f,
    x_range,
    y_range,
    mode="full",
    ax=None,
    scaling=default_scaling,
    verbose=False,
    levels=None,
)
```
The arguments are:
- `f`: a real or complex function of real or complex variable
- `x_range`: either a tuple (x_min, x_max, n_x) or an array-like object
- `y_range`: same as `x_range`
- `mode`: a string, recognised values are "real", "imag", "reim", "full", "abs", "arg", "3D"
- `ax`: a `matplotlib.Axes` instance
- `scaling`: a function that maps [0, +infnity) into [0, 1]
- `verbose`: true or false
- `levels`: int or array-like

If mode is one of ("real", "imag", "reim"), the script fills an array of values
for the independent variable, evaluates the `f` on it, and plots the result,
using the `Axes` instance if provided or generating one if not. If
`mode=="real"`, only the real part is plotted, `mode=="imag"`, only the
imaginary part is plotted, and if `mode==reim` both are plotted ("reim" is
short for "REal IMag"). All other arguments are ignored.

If mode is one of ("full", "abs", "arg", "3D"), things procede just like above,
except that a 2D grid of points in the complex plane is generated according to
`x_range` and `y_range` and the function is evaluated on that. The plot type depend on `mode`:
- if `mode==full`, domain coloring is used: luminosity represents magnitude and
  color represents argument
- if `mode==arg`, same as "full", but only the argument is shown. The magnitude
  is ignored and the luminosity is constant
- if `mode==abs`, only the magnitude is plotted (with a colormap), the argument
  is ignored
- if `mode==3D`, same as "full" , but the function is also represented as a
  surface in 3D space
If `verbose==true`, the scripts prints on how many points of the grid the
function has been evaluate so far (it can take a while for very fine grids). If
`mode!=abs`, the way the magnitude is plotted can be controlled by supplying a
custom scaling function ion argument `scaling`. If `mode==abs`, the plot will
contain isocontours of magnitude to guide the eye. If `levels==None`, these
will generated automatically, or they can supplied explicitly.

For the domain coloring algorithm, this script incorporates much code from
[https://github.com/nschloe/cplot].
