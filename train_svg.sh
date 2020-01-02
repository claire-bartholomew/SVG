#$ -cwd
#$ -V
#$ -l coproc_p100=1
#$ -l h_rt=48:00:00
#$ -m be

source /nobackup/sccsb/miniconda3/bin/activate /nobackup/sccsb/miniconda3/envs/svg_env

# Python script with all parameters
#python ./svg/generate_svg_lp.py --model_path ./svg/pretrained_models/svglp_bair.pth --log_dir ./svg/images --data_root /nobackup/sccsb/bair > log_svg.out
python train_svg_lp_test.py --dataset radar --beta 0.0001 --epoch_size 2772 --n_past 5 --n_future 25 --data_root /nobackup/scssb/radar 
