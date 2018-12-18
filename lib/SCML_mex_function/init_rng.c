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

/* simply initialize random number generator for MEX-files */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    srand(0);
}