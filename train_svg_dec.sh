#$ -cwd
#$ -V
#$ -l coproc_v100=1
#$ -l h_rt=48:00:00
#$ -m be

source /nobackup/sccsb/miniconda3/bin/activate /nobackup/sccsb/miniconda3/envs/svg_env
unset GOMP_CPU_AFFINITY KMP_AFFINITY
# Python script with all parameters
#python train_svg_lp_cuda.py --dataset bair --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /nobackup/sccsb/bair/ --log_dir logs
#python ./svg/generate_svg_lp.py --model_path ./svg/pretrained_models/svglp_bair.pth --log_dir ./svg/images --data_root /nobackup/sccsb/bair > log_svg.out
python train_svg_lp_dec_augment_wandb_tb.py --dataset radar --beta 0.0001 --epoch_size 2772 --n_past 3 --n_future 7 --data_root /nobackup/scssb/radar
