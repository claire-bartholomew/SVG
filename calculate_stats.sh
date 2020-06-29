#$ -cwd
#$ -V
#$ -l coproc_v100=1
#$ -l h_rt=48:00:00
#$ -m be

source /nobackup/sccsb/miniconda3/bin/activate /nobackup/sccsb/miniconda3/envs/svg_env

# Python script with all parameters
python calculate_data_stats.py
