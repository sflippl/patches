#!/bin/sh
#
#
#SBATCH --job-name=20-cbm-experiment    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=20:00              # The time the job will take to run.
 
cd experiments
python 20-cbm-comparison.py --iterations 20000
 
# End of script