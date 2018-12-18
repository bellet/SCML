# Sparse Compositional Metric Learning

SCML is a Matlab/MEX implementation of [Sparse Compositional Metric Learning](http://researchers.lille.inria.fr/abellet/papers/aaai14.pdf).
It allows scalable learning of global, multi-task and multiple local Mahalanobis metrics for multi-class data under a unified framework based on sparse combinations of rank-one basis metrics.

SCML is distributed under GNU/GPL 3 license.

## Getting started

Please run (inside the matlab console)
```
install
demo_global_local   % demo of SCML-Global and SCML-Local
demo_multi_task     % demo of mt-SCML
```

Note: this code uses LDA basis set (as in the paper), but it is easily modifiable so that the user may input its own basis set if desired.

## Reference

If you use this code in scientific work, please cite:

- Y. Shi, A. Bellet and F. Sha. *Sparse Compositional Metric Learning*. AAAI Conference on Artificial Intelligence (AAAI), 2014.

## Acknowledgments

- Our code borrows some helper functions from the packages of Large Margin Nearest Neighbor (LMNN) and Parametric Local Metric Learning (PLML).
- We thank Shreyas Saxena for his bug reports and his help on fixing them.

## Version history

- v1.11 (8/2/2016): minor bug fix in multi-task objective computation (thanks to Junjie Hu).
- v1.1 (8/16/2015): various minor bug fixes and improvements. The basis and triplet generation now fully supports with datasets with very small classes and arbitrary labels (no need to be consecutive or positive). The computational and memory efficiency of the code when data is high dimensional has been largely improved, and we generate a rectangular (smaller) projection matrix when the number of selected basis is smaller than the dimension. K-NN classification with local metrics has been optimized and made significantly less costly in both time and memory.
- v1 (5/26/2014): initial release.
