$ontext
   Convex Nonparametric Least Squares (CNLS) example

   by Andy Johnson June 2015

   Data from:
      Timo Kuosmanen
         Finnish Electricity Distribution Data

Calculates a production function and overall economic efficiency


$offtext


sets i       "DMU's"  /1*89/
     j       'inputs and outputs' /OPEX, CAPEX, TOTEX, Energy, Length, Customers, PerUndGr/
     inp(j)  'inputs'  /OPEX, CAPEX/
     outp(j) 'outputs' /Energy, Length, Customers/
     cost(j) 'cost' /TOTEX/
     z(j)    'z variables' /PerUndGr/
;



Table data(i,j)
$ Include "C:\Users\dais2\Dropbox (Aalto)\Finnish_electricity_firm.txt"
;

alias(i,h)
;

PARAMETERS
   x(i,inp)  'inputs of firm i'
   y(i) 'outputs of firm i'

   alphavalue(i)    'output alpha values'
   betavalue(i,inp) 'output beta values'
   residualvalue(i) 'output residual values'

;

x(i,inp) = data(i,inp);
y(i) = data(i,'Energy');

Variables  alpha(i)     intercept term
            beta(i,j)     input coefficients
            e(i)     error terms
            sse      sum of squared errors;

Positive Variables beta ;

Equations  obj      objective function
            err(i)   regression equation
            conv(i,h)   convexity ;


obj..  sse =e= sum(i, sqr(e(i)))  ;

err(i).. y(i) =e= alpha(i) + sum(inp, beta(i,inp)*x(i,inp)) + e(i) ;

conv(i,h).. alpha(i) + sum(inp, beta(i,inp)*x(i,inp)) =l= alpha(h) + sum(inp, beta(h,inp)*x(i,inp));

Model cnls model / all / ;

OPTION QCP=MOSEK;

Solve cnls using QCP minimizing sse ;

Display alpha.l,beta.l, e.l;

