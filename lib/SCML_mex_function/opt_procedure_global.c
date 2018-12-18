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

/*double ddot_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int max_iter, nBasis, iter, i, j, count, output_iter, sizeT, idx;
    double beta, gamma, best_obj, obj1, obj2, slack_val, scale_f;
    double *best_w, *obj, *nImp, *w, *avg_grad_w, *dist_diff;
    
    /*ptrdiff_t incm, incn, dim;*/
    
    if (nrhs != 8)
        mexErrMsgTxt("Function needs 8 arguments: {w, avg_grad_w, dist_diff, sizeT, beta, gamma, max_iter, output_iter}");
    
    /* Input */
    w = (double *)mxGetData(prhs[0]);  /* 1*d */
    avg_grad_w = (double *)mxGetData(prhs[1]);  /* 1*d */
    dist_diff = (double *)mxGetData(prhs[2]);  /* sizeT -by- nBasis*/
    sizeT = mxGetScalar(prhs[3]);
    beta = mxGetScalar(prhs[4]);
    gamma = mxGetScalar(prhs[5]);
    max_iter = mxGetScalar(prhs[6]);
    output_iter = mxGetScalar(prhs[7]);
    
    nBasis = mxGetN(prhs[0]);
    
    /* Output */
    plhs[0] = mxCreateNumericMatrix(1, nBasis, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(max_iter, 1, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(max_iter, 1, mxDOUBLE_CLASS, mxREAL);
    
    best_w   = (double *)mxGetData(plhs[0]);
    obj      = (double *)mxGetData(plhs[1]);
    nImp     = (double *)mxGetData(plhs[2]);
    
    /* core procedure */
    best_obj = 1e10;
    
    /*        
    dim = (ptrdiff_t)nBasis;
    incm = (ptrdiff_t)sizeT;
    incn = (ptrdiff_t)1;
     */
    
    for(iter = 1; iter <= max_iter; iter++) {
        if (iter%output_iter == 1) {
            
            obj1 = 0;
            for (i=0; i<nBasis; i++)
                obj1 = obj1 + w[i];
            obj1 = obj1 * beta;
            
            obj2 = 0;
            count = 0;
            for (i=0; i<sizeT; i++) {
                slack_val = 1;
                
                for (j=0; j<nBasis; j++)
                    slack_val += dist_diff[sizeT*j + i] * w[j];
  
                /*slack_val += ddot_(&dim, &dist_diff[i], &incm, w, &incn);*/
                
                if (slack_val > 0) {
                    count++;
                    obj2 += slack_val;
                }
            }
            obj2 = obj2 / sizeT;
            
            obj[iter-1] = obj1 + obj2;
            nImp[iter-1] = (double)count;
            
            printf("[Global] iter %d\t obj %.6f\t num_imp %d\n", iter, obj[iter-1], (int)nImp[iter-1]);
            
            /* update the best */
            if (obj[iter-1] < best_obj)
            {
                best_obj = obj[iter-1];
                for (j=0; j<nBasis; j++)
                    best_w[j] = w[j];
            }
        }
  
        idx = rand()%sizeT;
        
        slack_val = 1;
        
        /*    slack_val += ddot_(&dim, &dist_diff[idx], &incm, w, &incn); */
        
        for (j=0; j<nBasis; j++)
            slack_val += w[j]*dist_diff[sizeT*j + idx];
        
        
        if (slack_val > 0) {
            for (j=0; j<nBasis; j++) {
                avg_grad_w[j] = (iter-1) *  avg_grad_w[j] + dist_diff[sizeT*j + idx];
                avg_grad_w[j] /= iter;
            }
        }
        else {
            for (j=0; j<nBasis; j++) {
                avg_grad_w[j] = (iter-1) *  avg_grad_w[j];
                avg_grad_w[j] /= iter;
            }
        }
        
        scale_f = -sqrt(iter) / gamma;
        
        for (j=0; j<nBasis; j++) {
            if ( fabs(avg_grad_w[j]) < beta )
                w[j] = 0;
            else {
                if (avg_grad_w[j] > 0)
                    w[j] = scale_f * ( avg_grad_w[j] - beta );
                else
                    w[j] = scale_f * ( avg_grad_w[j] + beta );
                
                if (w[j] < 0)
                    w[j] = 0;
            }
        }
    }
    
    printf("max iteration reached.\n");
}
