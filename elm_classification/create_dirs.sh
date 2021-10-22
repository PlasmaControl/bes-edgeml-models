#!/bin/zsh

base_dir="outputs/ts_anomaly_detection_plots/"
sw_8='signal_window_8'
sw_16='signal_window_16'
sw_32='signal_window_32'
sw_64='signal_window_64'

for path in $base_dir$sw_8 $base_dir$sw_16 $base_dir$sw_32 $base_dir$sw_64
do
    cd $path
    /bin/mkdir -p 'label_look_ahead_0' 'label_look_ahead_50' 'label_look_ahead_100' 'label_look_ahead_150' 'label_look_ahead_200'
    cd ../../../
done
