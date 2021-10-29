#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn
from typing import Optional

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from centermask.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    Caffe2Tracer,
    TracingAdapter,
    add_export_config,
    dump_torchscript_IR,
    scripting_with_instances,
)

from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.backbone import backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

import numpy as np

def get_sample_inputs(sample_image, cfg):
    os.path.exists(sample_image)
    # get a sample data
    original_image = detection_utils.read_image(sample_image, format=cfg.INPUT.FORMAT)
    # Do same preprocessing as DefaultPredictor
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    inputs = {"image": image, "height": height, "width": width}

    sample_inputs = [inputs]
    return sample_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='v4_0094999_aug_color.pth')
    parser.add_argument('--onnx-file', '-onnx', type=str, default='model.onnx')
    parser.add_argument('--sample_image', type=str, default="test_imgs/jzx.jpg")
    
    args = parser.parse_args()
    
    config_file = args.config
    checkpoint_file = args.checkpoint
    saved_model = args.onnx_file
    sample_image = args.sample_image

    logger = setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint_file

    sample_inputs = get_sample_inputs(sample_image, cfg)

    # create a torch model
    torch_model = build_model(cfg)
    torch_model = torch_model.cuda()
    DetectionCheckpointer(torch_model).load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    assert TORCH_VERSION >= (1, 8)

    image = sample_inputs[0]["image"].cuda()
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    with PathManager.open(saved_model, "wb") as f:
        torch.onnx.export(traceable_model, image, f, verbose=True, opset_version=11,
            input_names=["input"],
            output_names=["pred_boxes", "pred_classes", "scores", "size"],
            dynamic_axes={
                "input": {1: "height", 2: "width"},
                "size": {0: "h", 1: "w"}
            }
        )