#!/usr/bin/env python3
"""Build a tiny stand-in for a Perch v2 model.

Matches the real model's contract as the birdnet-onnx crate detects it: one
float32 input of 160,000 samples, and four outputs in the published order
(embedding, spatial_embedding, spectrogram, label), with static trailing
dimensions. The class scores are the constants 2.0, 1.0 and 0.0 whatever the
input, so the softmax rest logit wiring can be exercised against known logits
without downloading a 65 MB model.

Run once; the .onnx output is committed.

Usage:
    python3 tests/fixtures/make_fixture_perch.py
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

SAMPLES = 160_000
SPECIES = 3
EMBEDDING = 2

# `label = m * W + B` with W = 0, so the logits are the bias for any input.
WEIGHTS = np.zeros((1, SPECIES), dtype=np.float32)
BIAS = np.array([2.0, 1.0, 0.0], dtype=np.float32)
EMBEDDING_WEIGHTS = np.ones((1, EMBEDDING), dtype=np.float32)
EMBEDDING_BIAS = np.zeros(EMBEDDING, dtype=np.float32)

graph = helper.make_graph(
    [
        helper.make_node("ReduceMean", ["input"], ["m"], axes=[1], keepdims=1),
        helper.make_node("Gemm", ["m", "W", "B"], ["label"]),
        helper.make_node("Gemm", ["m", "We", "Be"], ["embedding"]),
        helper.make_node("Reshape", ["embedding", "spatial_shape"], ["spatial_embedding"]),
        helper.make_node("Reshape", ["m", "spectrogram_shape"], ["spectrogram"]),
    ],
    "fixture_perch",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", SAMPLES])],
    [
        helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", EMBEDDING]),
        helper.make_tensor_value_info(
            "spatial_embedding", TensorProto.FLOAT, ["batch", 1, 1, EMBEDDING]
        ),
        helper.make_tensor_value_info("spectrogram", TensorProto.FLOAT, ["batch", 1, 1]),
        helper.make_tensor_value_info("label", TensorProto.FLOAT, ["batch", SPECIES]),
    ],
    [
        helper.make_tensor("W", TensorProto.FLOAT, [1, SPECIES], WEIGHTS.flatten()),
        helper.make_tensor("B", TensorProto.FLOAT, [SPECIES], BIAS),
        helper.make_tensor("We", TensorProto.FLOAT, [1, EMBEDDING], EMBEDDING_WEIGHTS.flatten()),
        helper.make_tensor("Be", TensorProto.FLOAT, [EMBEDDING], EMBEDDING_BIAS),
        helper.make_tensor("spatial_shape", TensorProto.INT64, [4], [-1, 1, 1, EMBEDDING]),
        helper.make_tensor("spectrogram_shape", TensorProto.INT64, [3], [-1, 1, 1]),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, "tests/fixtures/fixture-perch.onnx")
print("wrote tests/fixtures/fixture-perch.onnx")
