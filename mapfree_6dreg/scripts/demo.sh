#!/bin/bash
python demo.py \
    config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    --checkpoint pretrained_models/far_loftr.ckpt \
    --output tranform_mapfree.json
