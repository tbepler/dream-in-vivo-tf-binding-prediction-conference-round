#BSUB -q priority
#BSUB -n 12
#BSUB -R rusage[mem=1500]
#BSUB -W 172:00
#BSUB -J 01_resid_conv_logit_cross_validate
#BSUB -eo experiments/01_resid_conv_logit_cross_validate/lsf.err
#BSUB -oo experiments/01_resid_conv_logit_cross_validate/lsf.out

#source activate onlydreams
export OMP_NUM_THREADS=12
python -u experiments/01_resid_conv_logit_cross_validate.py
