CUDA_VISIBLE_DEVICES=0 python demo.py \
--center-num 48 \
--embed-dim 256 \
--patch-size 64 \
--checkpoint './checkpoints/realsense' \
--rgb-path './images/demo_rgb.png' \
--depth-path './images/demo_depth.png'
