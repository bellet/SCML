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
    double *W, *avg_grad_W, *norm_col, *best_W, *obj, *nImp, *sizeT_set, *grad_W_task, *prev_avg_grad;
    double beta, gamma, slack_var, best_obj, *tmp, q1, q2, q3, scale_f, max_val, cur_obj, cur_obj_triplet, norm_W_col, sizeT_tol;
    int max_iter, output_iter, iter, i, j, nTask, nBasis, id_task, idx, sizeT_task, t, cur_n_imp;
    mxArray *d_task;
	
    if (nrhs != 9)
        mexErrMsgTxt("Function needs 9 arguments: {W, avg_grad_W, dist_diff_cell, norm_col, sizeT_set, beta, gamma, MAX_ITER, output_iter}");

    /* Input */
    W = (double *)mxGetData(prhs[0]);  /* nTask * nBasis */
    avg_grad_W = (double *)mxGetData(prhs[1]);  /* nTask * nBasis */
    
    nTask = mxGetM(prhs[0]);
    nBasis = mxGetN(prhs[0]);
    
    norm_col = (double *)mxGetData(prhs[3]);  /* nBasis vector */
    sizeT_set = (double *)mxGetData(prhs[4]);   /* nTask vector */
    
    sizeT_tol = 0;
    for (t=0; t<nTask; t++)
        sizeT_tol += sizeT_set[t];
    
	beta = mxGetScalar(prhs[5]);
	gamma = mxGetScalar(prhs[6]);
    
	max_iter = mxGetScalar(prhs[7]);
	output_iter = mxGetScalar(prhs[8]);
	
    /* Output */
	plhs[0] = mxCreateNumericMatrix(nTask, nBasis, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(max_iter, 1, mxDOUBLE_CLASS, mxREAL);
	plhs[2] = mxCreateNumericMatrix(max_iter, 1, mxDOUBLE_CLASS, mxREAL);
    
	best_W   = (double *)mxGetData(plhs[0]);
    obj      = (double *)mxGetData(plhs[1]);
	nImp     = (double *)mxGetData(plhs[2]);   
    
    grad_W_task = (double *)calloc(nBasis, sizeof(double));
    prev_avg_grad = (double *)calloc(nBasis, sizeof(double));
    
	/* core procedure */
	best_obj = 1e10;
	
    for(iter = 1; iter <= max_iter; iter++) 
    {
        if (iter%output_iter == 1) 
        {
            cur_obj = 0;
            
            for (j=0; j<nBasis; j++)
            {
                norm_W_col = 0;
                for (t=0; t<nTask; t++)
                    norm_W_col += W[nTask*j + t]*W[nTask*j + t];
                cur_obj += beta*sqrt(norm_W_col);
            }
            
            cur_n_imp = 0;
            
            cur_obj_triplet = 0;
            
            for (t=0; t<nTask; t++)
            {
                d_task = mxGetCell(prhs[2], t);
                tmp = (double *)mxGetData(d_task);

                for(i=0; i<sizeT_set[t]; i++)
                {
                    slack_var = 1;
                    
                    for(j=0; j<nBasis; j++)
                        slack_var += tmp[ (int)sizeT_set[t]*j + i] * W[ nTask*j + t];
                    
                    if (slack_var > 0)
                    {
                        cur_obj_triplet += slack_var;
                        cur_n_imp++;
                    }
                }
            }
            
            cur_obj += cur_obj_triplet / sizeT_tol;
            
            obj[iter-1] = cur_obj;
            nImp[iter-1] = cur_n_imp;
            
            printf("[MT] iter %d\t obj %.6f\t num_imp %d\n", iter, obj[iter-1], (int)nImp[iter-1]);
            
            if (obj[iter-1] < best_obj)
            {
                best_obj = obj[iter-1];
                for (t=0; t<nTask; t++)
                    for (j=0; j<nBasis; j++)
                        best_W[nTask*j + t] = W[nTask*j + t];
            }
        }
        
        id_task = rand()%nTask;
        idx = rand()%  (int)sizeT_set[id_task];
        
        slack_var = 1;
        
        d_task = mxGetCell(prhs[2], id_task);
        tmp = (double *)mxGetData(d_task);
        
        sizeT_task = (int)sizeT_set[id_task];
            
        for (j = 0; j<nBasis; j++)
        {
            slack_var += tmp[ sizeT_task*j + idx] * W[nTask*j + id_task];
        }
        
        if (slack_var > 0)
        {
            for (j=0; j<nBasis; j++)
                grad_W_task[j] = tmp[ sizeT_task*j + idx];
        }
        else
        {
            for (j=0; j<nBasis; j++)
                grad_W_task[j] = 0;
        }
        
        scale_f = (double)(iter-1)/iter;
        
        for (j=0; j<nBasis; j++)
        {
            prev_avg_grad[j] = avg_grad_W[nTask*j + id_task];
            
            for (t = 0; t<nTask; t++)
            {
                if (t != id_task)
                    avg_grad_W[ nTask*j + t] = scale_f * avg_grad_W[ nTask*j + t];
                else
                    avg_grad_W[ nTask*j + t] = scale_f * avg_grad_W[ nTask*j + t] + grad_W_task[j]/iter;
            }
        }
        
        /* update variables */
        
        for (j=0; j<nBasis; j++)
        {           
            q1 = norm_col[j] * scale_f;
            q2 = prev_avg_grad[j] * scale_f;
            q3 = grad_W_task[j]/iter + prev_avg_grad[j] * scale_f;
            
            norm_col[j] = sqrt( q1*q1 - q2*q2 + q3*q3 );
            
            for (t=0; t<nTask; t++)
            {
                max_val = 1 - beta/norm_col[j];
                if (max_val < 0)
                    W[nTask*j + t] = 0;
                else
                    W[nTask*j + t] = (-sqrt(iter)/gamma) * max_val * avg_grad_W[ nTask*j + t];
                
                if (W[nTask*j + t] < 0)
                    W[nTask*j + t] = 0;
            }
        }
    }
  
	printf("max Iteration reached.\n");  
}