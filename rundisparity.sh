#!/bin/bash
python disparity.py --maxdisp 192  \
    --cnn_imgs_path /Ensembler/mount/cnn/ \
    --crf_imgs_path /Ensembler/mount/crf/ \
    --disp_L_path /Ensembler/training/disp_occ_0 \
    --savedisp /Ensembler/mount/
