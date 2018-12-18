/*
 *
 * input: x,a,b,M
 *    x : matrix d x n
 *    a : vector 1 x number of triplet constraints
 *    b : vector 1 x number of triplet constraints
 *    W : linear wegiht vector 1 x n  
 *
 * 
 *
 */

#include "mex.h"
#include <string.h>

/* If you are using a compiler that equates NaN to zero, you must
 * compile this example using the flag -DNAN_EQUALS_ZERO. For
 * example:
 *
 *     mex -DNAN_EQUALS_ZERO findnz.c
 *
 * This will correctly define the IsNonZero macro for your
   compiler. */

#if NAN_EQUALS_ZERO
#define IsNonZero(d) ((d) != 0.0 || mxIsNaN(d))
#else
#define IsNonZero(d) ((d) != 0.0)
#endif


double square(double x) 
{  return(x*x); 
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int nos_constr;
  int n, m, i, j, r, c;
  double *X, *dummy, *v1,*v2, *C, *out,*W;
  double *av,*bv, *M;
  double sum;


  /* Check for proper number of input and output arguments. */
  if (nrhs != 4) {
    mexErrMsgTxt("Exactly four input arguments required.");
  }

  if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  /* Check data type of input argument. */
  if (!(mxIsDouble(prhs[0]))) {
   mexErrMsgTxt("Input array must be of type double.");
  }



  /* Get the number of elements in the input argument. */
  nos_constr = mxGetNumberOfElements(prhs[1]);
  if(nos_constr != mxGetNumberOfElements(prhs[2]))
    mexErrMsgTxt("Hey Bongo! Both index vectors must have same length!\n");
  n = mxGetN(prhs[0]);
  m = mxGetM(prhs[0]);

  /* Get the data. */
  X   = mxGetPr(prhs[0]);
  av  = mxGetPr(prhs[1]);
  bv  = mxGetPr(prhs[2]);
   W = mxGetPr(prhs[3]);
  /* Create output matrix */
  plhs[0] = mxCreateDoubleMatrix( nos_constr, 1, mxREAL);
  out = mxGetPr(plhs[0]);

  dummy = malloc( m*sizeof(double) );
  C     = malloc( m*m*sizeof(double) );

 /* compute outer products and sum them up */
  for( i=0; i<nos_constr; i++ )
  {

   /* Assign cols addresses */
   v1 = &X[(int) (av[i]-1)*m];
   v2 = &X[(int) (bv[i]-1)*m];

   for(j=0;j<m;j++)
   {  dummy[j] = v1[j] - v2[j];}
   sum=0;

   for(r=0;r<m;r++)
   {
	
      sum =sum+W[(int)av[i]-1]*dummy[r]*dummy[r];
   }
    out[i] = sum;
  }


  free(dummy);
  free( C );
}



