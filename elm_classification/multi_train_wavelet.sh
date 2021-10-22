#!/bin/zsh

# conda activate torch
echo "Running first training round"
python lstm_autoencoder.py --model_name lstm_ae --data_preproc wavelet --signal_window_size 64 --label_look_ahead 0 --hidden_size 32 --device cuda --batch_size 256 --n_epochs 60 --max_elms -1 --normalize_data --truncate_inputs --filename_suffix _wavelet
echo "Running second training round"
python lstm_autoencoder.py --model_name lstm_ae --data_preproc wavelet --signal_window_size 64 --label_look_ahead 50 --hidden_size 32 --device cuda --batch_size 256 --n_epochs 60 --max_elms -1 --normalize_data --truncate_inputs --filename_suffix _wavelet
echo "Running third training round"
python lstm_autoencoder.py --model_name lstm_ae --data_preproc wavelet --signal_window_size 64 --label_look_ahead 100 --hidden_size 32 --device cuda --batch_size 256 --n_epochs 60 --max_elms -1 --normalize_data --truncate_inputs --filename_suffix _wavelet
echo "Running fourth training round"
python lstm_autoencoder.py --model_name lstm_ae --data_preproc wavelet --signal_window_size 64 --label_look_ahead 150 --hidden_size 32 --device cuda --batch_size 256 --n_epochs 60 --max_elms -1 --normalize_data --truncate_inputs --filename_suffix _wavelet
echo "Running fifth training round"
python lstm_autoencoder.py --model_name lstm_ae --data_preproc wavelet --signal_window_size 64 --label_look_ahead 200 --hidden_size 32 --device cuda --batch_size 256 --n_epochs 60 --max_elms -1 --normalize_data --truncate_inputs --filename_suffix _wavelet