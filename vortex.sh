#!/bin/bash
#SBATCH --job-name=KRMI
#SBATCH --time=14:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --account=cbronze
#SBATCH --partition=pbatch
#SBATCH -o KRMI-%j

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=gibson48@llnl.gov

## launch the calulation run
files=$(ls pickles/)
for file in ${files[@]}
do
	mkdir -p output/${file%.*}
	time srun -N 1 -n 16 python KNN_train.py "${file}" > "output/${file%.*}/log.out" 2> "output/${file%.*}/log.err" &
done
wait