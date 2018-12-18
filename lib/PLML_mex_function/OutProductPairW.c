/*
 *
 * input: x,T
 *    x : matrix d x n
 *    SS : target neighbor matrix  2* number of target neighbor relationship
 *    W:  linear weight matrix   n* number of anchor points
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
  
  int n, d, numPair, i,k,j,jj;
  double *X, *P,*W, *out,*dummy1,*dummy2,*v1,*v2,*v3;
 
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
  numPair=mxGetN(prhs[1]);

  /* Get the data. */
  X   = mxGetPr(prhs[0]);
  P  = mxGetPr(prhs[1]);
  W   = mxGetPr(prhs[2]);
  /* Create output matrix */
  plhs[0] = mxCreateDoubleMatrix( 1,d*d, mxREAL);
  out = mxGetPr(plhs[0]);

     dummy1 = malloc( d*sizeof(double) );
     

 /* compute outer products and sum them up */
  for( i=0; i<numPair; i++ )
   {
      v1 = &X[(int) (P[i*2]-1)*d];
      v2 = &X[(int) (P[i*2+1]-1)*d];
      
    for(jj=0;jj<d;jj++)
   {  dummy1[jj] = v1[jj] - v2[jj];}
      
 
     for(j=0;j<d;j++)
     {
        for (k=0;k<d;k++)
        {
            double sum=dummy1[j]*dummy1[k];
            out[j*d+k] = out[j*d+k]+sum*W[(int)P[i*2]-1];
        }
     }
  }

}



