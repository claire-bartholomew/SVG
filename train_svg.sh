#$ -cwd
#$ -V
#$ -l coproc_p100=1
#$ -l h_rt=48:00:00
#$ -m be

source /nobackup/sccsb/miniconda3/bin/activate /nobackup/sccsb/miniconda3/envs/svg_env

# Python script with all parameters
#python ./svg/generate_svg_lp.py --model_path ./svg/pretrained_models/svglp_bair.pth --log_dir ./svg/images --data_root /nobackup/sccsb/bair > log_svg.out
python train_svg_lp_test.py --dataset radar --model vgg --beta 0.0001 --epoch_size 2772 --n_past 3 --n_future 7 --data_root /nobackup/scssb/radar 
#python train_svg_lp_test_moredata.py --dataset radar --beta 0.0001 --epoch_size 10800 --data_root /nobackup/scssb/radar #--model_dir /home/home01/sccsb/SVG/logs/lp/radar/model=dcgan128x128-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=3-n_future=7-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=True-beta=0.0001000
