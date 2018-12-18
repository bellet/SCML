/*
 *
 * input: x,T
 *    x : matrix d x n
 *    T : triplet constraint matrix  3 * number of triplet constraints
 *    U : dual parameter matrix 1 * number of triplet constraints 
 *
 * output: vec(outerProduct) of each triplet
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




void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  
  int n, d, numTriplet, i,k,j,jj;
  double *X, *T,*U, *out,*dummy1,*dummy2,*v1,*v2,*v3;
 
  double sum;


  /* Check for proper number of input and output arguments. */
  if (nrhs != 3) {
    mexErrMsgTxt("Exactly three input arguments required.");
  }

  if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  /* Check data type of input argument. */
  if (!(mxIsDouble(prhs[0]))) {
   mexErrMsgTxt("Input array must be of type double.");
  }



  n = mxGetN(prhs[0]);
  d = mxGetM(prhs[0]);
  numTriplet=mxGetN(prhs[1]);

  /* Get the data. */
  X   = mxGetPr(prhs[0]);
  T  = mxGetPr(prhs[1]);
  U   = mxGetPr(prhs[2]);
  /* Create output matrix */
  plhs[0] = mxCreateDoubleMatrix( 1,d*d, mxREAL);
  out = mxGetPr(plhs[0]);

     dummy1 = malloc( d*sizeof(double) );
      dummy2 = malloc( d*sizeof(double) );

 /* compute outer products and sum them up */
  for( i=0; i<numTriplet; i++ )
   {
      v1 = &X[(int) (T[i*3]-1)*d];
      v2 = &X[(int) (T[i*3+1]-1)*d];
      v3 = &X[(int) (T[i*3+2]-1)*d];
      
    for(jj=0;jj<d;jj++)
   {  dummy1[jj] = v1[jj] - v2[jj];}
       for(jj=0;jj<d;jj++)
   {  dummy2[jj] = v1[jj] - v3[jj];}
      
 
     for(j=0;j<d;j++)
     {
        for (k=0;k<d;k++)
        {
            double sum=dummy2[j]*dummy2[k]-dummy1[j]*dummy1[k];
            out[j*d+k] = out[j*d+k]+sum*U[i];
        }
     }
  }

}



