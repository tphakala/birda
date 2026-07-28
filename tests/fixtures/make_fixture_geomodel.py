#!/usr/bin/env python3
"""Build a tiny stand-in for the BirdNET Geomodel.

Matches the real model's input and output contract: 3 float32 inputs
(latitude, longitude, week) and N sigmoid outputs, one per species. Used so the
geomodel code path can be exercised in CI without downloading 14.7 MB.

Run once; the .onnx output is committed.

Usage:
    python3 tests/fixtures/make_fixture_geomodel.py
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

# Five species. Weights chosen so a mid-latitude query yields a spread of
# scores across the threshold, rather than all-high or all-low.
WEIGHTS = np.array(
    [
        [0.010, -0.020, 0.030, 0.001, 0.050],
        [0.005, 0.010, -0.015, 0.002, 0.020],
        [0.100, 0.050, -0.200, 0.010, 0.150],
    ],
    dtype=np.float32,
)
BIAS = np.array([0.5, -3.0, 0.2, -9.0, 1.0], dtype=np.float32)

graph = helper.make_graph(
    [
        helper.make_node("Gemm", ["input", "W", "B"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["probabilities"]),
    ],
    "fixture_geomodel",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3])],
    [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 5])],
    [
        helper.make_tensor("W", TensorProto.FLOAT, [3, 5], WEIGHTS.flatten()),
        helper.make_tensor("B", TensorProto.FLOAT, [5], BIAS),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, "tests/fixtures/fixture-geomodel.onnx")
print("wrote tests/fixtures/fixture-geomodel.onnx")
