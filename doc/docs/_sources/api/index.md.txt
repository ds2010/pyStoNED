## API Documentation

### The CNLS Class    

`class CNLS.CNLSy, x, z=None, cet='addi', fun='prod', rts='vrs')`

Returns the Convex Nonparametric Least Square (CNLS) estimates with a specific function, error term, and returns to scale. This function is the fundamental  class in the entire pyStoNED package.

The CNLS class syntax has the following arguments:

    y Required. Output variable.
    x Required. Input variable.
    z Optional. Contextual variable.
    cet Required. crt = "addi", additive composite error term; 
                  crt = "mult", multiplicative composite error term.
    fun Required. fun = "prod", production frontier; 
                  fun = "cost", cost frontier.
    rts Required. rts = "vrs", variable returns to scale; 
                  rts = "crs", constant returns to scale.


### stoned() function

Returns the technical inefficiency with a specific function, decomposition methods, and error term. This function is used to decompose the residual and also is the fundamental function in the entire pyStoNED package.

Syntax

stoned(y, eps, func, method, crt)

The stoned() function syntax has the following arguments:

y Required. Output variable.
eps Required. Residuals estimated by function cnls() or other functions such as cnlsz().
func Required. func = "prod", production frontier; func = "cost", cost frontier.
method Required. method = "MOM", residual decomposition by method of moments; method = "QLE", residual decomposition by quasi-likelihood estimation; method = "NKD", residual decomposition by Nonparametric kernel deconvolution.
crt Required. crt = "addi", additive composite error term; crt = "mult", multiplicative composite error term.



### icnls() function

Returns the Isotonic Convex Nonparametric Least Square estimates with a specific function, error term, and returns to scale. This function is the convexity relaxed CNLS() and can be directly compared with FDH estimator.

Syntax

icnls(y, x, p, crt, func, pps)

The icnls() function syntax has the following arguments:

y Required. Output variable.
x Required. Input variable.
crt Required. crt = "addi", additive composite error term; crt = "mult", multiplicative composite error term.
func Required. func = "prod", production frontier; func = "cost", cost frontier.
pps Required. pps = "vrs", variable returns to scale; pps = "crs", constant returns to scale.

### ceqr() function

Returns the Convex quantile/expectile regression (CQR and CER) estimates with a specific function, error term, and returns to scale.

Syntax

cqer(y, x, tau, crt, func, pps, tile)

The ceqr() function syntax has the following arguments:

y Required. Output variable.
x Required. Input variable.
tau Required. Scalar value (e.g., tau = 0.5).
crt Required. crt = "addi", additive composite error term; crt = "mult", multiplicative composite error term.
func Required. func = "prod", production frontier; func = "cost", cost frontier.
pps Required. pps = "vrs", variable returns to scale; pps = "crs", constant returns to scale.
tile Required. tile = "quantile", quantile regression; tile = "expectile", expectile regression.

### dea() function

Returns the radial Data Envolpment Anaylsis (DEA) model estimates.

Syntax

dea(y, x, orient, rts)

The dea() function syntax has the following arguments:

y Required. Output variable.
x Required. Input variable.
orient Required. orient="io", input orientation; orient="oo", output orientation.
rts Required. rts="vrs", variable returns to scale; rts="crs", constant returns to scale.

### deaddf() function

Returns the directional Data Envolpment Anaylsis (DEA) model (DEA-DDF) estimates.

Syntax

deaddf(y, x, gx, gy, rts)

The dea() function syntax has the following arguments:

y Required. Output variable.
x Required. Input variable.
gx Required. Input vector (e.g., gx =1.0, 1.0).
gy Required. Output vector (e.g., gy = 0.0 or gy = 0.0, 0.0).
rts Required. rts="vrs", variable returns to scale; rts="crs", constant returns to scale.


### dea2cnls() function

Returns the first stage of Corrected Convex Nonparametric Least Squares (C2NLS) estimates.

Syntax

dea2cnls(y, x)

The dea2cnls() function syntax has the following arguments:

y Required. Output variable.
x Required. Input variable.


### ccnls() function

Returns the sencond stage of Corrected Convex Nonparametric Least Squares (C2NLS) estimates, including the adjusted residual and the constant term.

Syntax

ccnls2(eps, alpha)

The ccnls2() function syntax has the following arguments:

+ **eps**  Required. Residual estimated by `ccnls()` function.
+ **alpha**  Required. Constant term estimated by `ccnls()` function.


### cnlsddf() function

Returns the directional distance function estimates without undesirable outputs.

Syntax

cnlsddf(y, x, func, gx, gy)

The cnlsddf() function syntax has the following arguments:

+ **y**  Required. Output variable.
+ **x**  Required. Input variable.
+ **func**  Required. func = "prod", production frontier; func = "cost", cost frontier.
+ **gx**  Required. Intput vector (e.g., gx =[1.0, 1.0]).
+ **gy**  Required. Output vector (e.g., gy = 0.0 or gy = [0.0, 0.0]).




### cnlsddfb() function

Returns the directional distance function estimates with undesirable outputs.

Syntax

cnlsddfb(y, x, b, func, gx, gb, gy)

The cnlsddfb() function syntax has the following arguments:

+ **y**  Required. Output variable.
+ **x**  Required. Input variable.
+ **b**  Required. Undesirable output variable.
+ **func**  Required. func = "prod", production frontier; func = "cost", cost frontier.
+ **gx**  Required. Undesirable output vector (e.g., gx = [1.0, 1.0]).
+ **gb**  Required. Output vector (e.g., gb = 0.0 or gb = [0.0, 0.0]).
+ **gy**  Required. Output vector (e.g., gy = 0.0 or gy = [0.0, 0.0]).