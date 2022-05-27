import torch
import numpy as np

def to_onnx(model, c, w, h, onnx_name):
    dummy_input = torch.randn(1, c, w, h, device='cpu')
    torch.onnx.export(model, dummy_input, onnx_name, verbose=True)