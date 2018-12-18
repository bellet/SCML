fprintf('Installing...\n');
cur_dir = pwd;

cd('lib/PLML_mex_function');
mex('inner_prodW.c');
mex('OutProductInd.c');
mex('OutProductPairW.c');
mex('mink.c');
cd(cur_dir);

cd('lib/SCML_mex_function');
mex('opt_procedure_global.c');
mex('opt_procedure_local.c');
mex('opt_procedure_mt.c');
mex('init_rng.c');
cd(cur_dir);