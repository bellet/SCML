/*
Copyright 2014 Yuan Shi & Aurelien Bellet
 
This file is part of SCML.

SCML is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SCML is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SCML.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "mex.h"
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include <float.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    double *A, *b, *dist_diff, *zTr, *T, *best_A, *best_b, *obj, *nImp, *z_map, *tt, *weight, *norm_val;
    double beta, stepsize, slack_var, cur_obj, cur_obj_margin, scale, best_obj, beta2, alpha;
    int max_iter, output_iter, nBasis, sizeT, nZ, dZ, cur_n_imp, idx, idx2, iter, i, j, t;
    ptrdiff_t miniNum, row_Z, col_A, col_Z;
    char *paramDgemm = "N";
    
    miniNum = (ptrdiff_t)1;
    beta2 = 0;
    alpha = 1;
    
    if (nrhs != 9)
        mexErrMsgTxt("Function needs 9 arguments: {A, b, dist_diff, zTr, T, beta, stepsize, MAX_ITER, output_iter}");
    
    /* Input */
    A = (double *)mxGetData(prhs[0]);  /* dim_Z x nBasis */
    b = (double *)mxGetData(prhs[1]);  /* nBasis */
    dist_diff = (double *)mxGetData(prhs[2]);      /* sizeT x nBasis */
    zTr = (double *)mxGetData(prhs[3]);     /* nT x dZ */
    T = (double *)mxGetData(prhs[4]);  /* nTriplet x 3 */
    beta = mxGetScalar(prhs[5]);
    stepsize = mxGetScalar(prhs[6]);
    max_iter = mxGetScalar(prhs[7]);
    output_iter = mxGetScalar(prhs[8]);
    
    nZ = mxGetM(prhs[3]);
    row_Z = (ptrdiff_t)nZ;
    dZ = mxGetN(prhs[3]);
    col_Z = (ptrdiff_t)dZ;
    nBasis = mxGetN(prhs[0]);
    col_A = (ptrdiff_t)nBasis;
    
    sizeT = mxGetM(prhs[4]);
    
    
    /* Output */
    plhs[0] = mxCreateNumericMatrix(dZ, nBasis, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, nBasis, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(max_iter, 1, mxDOUBLE_CLASS, mxREAL);
    plhs[3] = mxCreateNumericMatrix(max_iter, 1, mxDOUBLE_CLASS, mxREAL);
    
    
    best_A   = (double *)mxGetData(plhs[0]);
    best_b   = (double *)mxGetData(plhs[1]);
    obj      = (double *)mxGetData(plhs[2]);
    nImp     = (double *)mxGetData(plhs[3]);
    
    /* core procedure */
    best_obj = 1e10;
    
    
    z_map = (double *)calloc(nBasis, sizeof(double));
    norm_val = (double *)calloc(nBasis, sizeof(double));
    tt = (double *)calloc(nBasis, sizeof(double));
    weight = (double *)calloc(nZ*nBasis, sizeof(double));
    
    for(iter = 1; iter <= max_iter; iter++) {
        
        /* proximal operator */
        if (iter%5000 == 1) 
        {
            if (beta > 0) {
                for (j=0; j<nBasis; j++) {
                    norm_val[j] = b[j]*b[j];
                    for (i=0; i<dZ; i++)
                        norm_val[j] += A[dZ*j + i]*A[dZ*j + i];
                }
                for (j=0; j<nBasis; j++) {
                    scale = 1 - 5000 * stepsize * beta / norm_val[j];
                    if (scale < 0) {
                        for (i=0; i<dZ; i++)
                            A[dZ*j + i] = 0;
                        b[j] = 0;
                    }
                    else {
                        for (i=0; i<dZ; i++)
                            A[dZ*j + i] *= scale;
                        b[j] *= scale;
                    }
                }
            }
        }
        
        if (iter%output_iter == 1) {
            
            cur_obj = 0;
            for (j=0; j<nBasis; j++) {
                norm_val[j] = b[j]*b[j];
                for (i=0; i<dZ; i++)
                    norm_val[j] += A[dZ*j + i]*A[dZ*j + i];
                
                cur_obj += beta * sqrt( norm_val[j] );
            }
            
            cur_n_imp = 0;

            /*      dgemm_(paramDgemm, paramDgemm, &row_Z, &col_A, &col_Z, &alpha, zTr, &row_Z, A, &col_Z, &beta2, weight, &row_Z); */    
            
            for (i=0; i<nZ; i++) {
                for (j=0; j<nBasis; j++) {
                    weight[ nZ*j + i ] = 0;
                    for (t=0; t<dZ; t++)
                        weight[ nZ*j + i ] += zTr[ nZ*t + i ] * A[ dZ*j + t ];
                    weight[ nZ*j + i ] += b[j];
                    
                    weight[ nZ*j + i ] *= weight[ nZ*j + i ];
                }
            }
            
            
            cur_obj_margin = 0;
            for (i=0; i<sizeT; i++) {
                slack_var = 1;
                idx2 = ( (int)T[i] ) - 1;
                for (j=0; j<nBasis; j++) {
                    slack_var += dist_diff[ sizeT*j + i] *  weight[ nZ*j + idx2 ];
                }
                
                if (slack_var > 0) {
                    cur_obj_margin += slack_var;
                    cur_n_imp++;
                }
            }
            
            cur_obj_margin /= sizeT;
            cur_obj += cur_obj_margin;
            
            obj[iter-1] = cur_obj;
            nImp[iter-1] = cur_n_imp;
            
            printf("[Local] iter %d\t obj %f\t num_imp %d\t step %f\n", iter, obj[iter-1],  (int)nImp[iter-1], stepsize);
            
            if (obj[iter-1] < best_obj) {
                best_obj = obj[iter-1];
                for (i=0; i<dZ; i++)
                    for (j=0; j<nBasis; j++)
                        best_A[dZ*j + i] = A[dZ*j + i];
                
                for (j=0; j<nBasis; j++)
                    best_b[j] = b[j];
            }
            
        }
        
        
        idx = rand()%sizeT;
        slack_var = 1;
        
        idx2 = (int)T[idx] - 1;
        
        for (j=0; j<nBasis; j++) {
            z_map[j] = 0;
            
            for (i=0; i<dZ; i++)
                z_map[j] += zTr[ nZ*i + idx2 ] * A[ dZ*j + i];
            
            z_map[j] += b[j];
            
            slack_var += z_map[j] * z_map[j] * dist_diff[ sizeT*j + idx ];
        }
        
        if (slack_var > 0) {
            for (j=0; j<nBasis; j++) {
                tt[j] = 2 * z_map[j] * dist_diff[ sizeT*j + idx ];
                b[j] -= stepsize * tt[j];
            }
            
            for (i=0; i<dZ; i++) {
                for (j=0; j<nBasis; j++)
                    A[ dZ*j + i] -= stepsize * zTr[ nZ*i + idx2 ] * tt[j];
            }
        }
        
        
    }
    
    printf("max iteration reached.\n");
    
    free(z_map);
    free(norm_val);
    free(tt);
    free(weight);
}
