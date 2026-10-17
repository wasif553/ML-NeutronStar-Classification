#!/bin/bash
# secondary experiments, run at reduced parallelism while the alternative population is being generated
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /c/w/pop
OUT_SUFFIX=_fix python pop_experiment.py "SO@2.0,SO@4.0" 7 > q_fix.log 2>&1
cd /c/w/ns && ONLY_SCALES=2.0,4.0 ONLY_FS=SO MAIN_OUT=results/main_fix.csv CONF_OUT=results/confusions_fix.json python main.py > q_legacy_fix.log 2>&1
cd /c/w/pop && python frozen_test.py 24 7 > q_frozen.log 2>&1
MCHI_MAX=938.783 N_PER_CLASS=63 OUT_SUFFIX=_mchi python pop_experiment.py "ALL@0,SO@0,NICER,NICER+f20,SO@1.0,SO@0.25" 7 > q_mchi.log 2>&1
python stat_robust.py 7 > q_stat.log 2>&1
echo ALLDONE > q_done.flag
