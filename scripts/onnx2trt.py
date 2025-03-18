import os
from glob import glob

import tensorrt as trt

__author__ = "YueLin"


def onnx2trt(onnx: str, half: bool = False) -> None:
    logger = trt.Logger()
    with trt.Builder(logger) as builder, builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, logger) as parser:
        with open(onnx, "rb") as model:
            if not parser.parse(model.read()):
                print("Error: onnx parse failed")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                    raise SystemExit(
                        "Failed to build the TensorRT engine!"
                    )
        print("Building an engine. This would take a while...")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        if half:
            config.flags |= 1 << int(trt.BuilderFlag.FP16)
        engine = builder.build_serialized_network(network, config)
    path = onnx.replace(".onnx", ".trt")
    with open(path, "wb") as f:
        f.write(engine)
    print("Serialized the TensorRT engine to file:", path)


def main(models: tuple) -> None:
    path = os.path.join(os.path.split(
        os.path.split(__file__)[0]
    )[0], "src", "server", "models")
    for model in map(lambda folder: os.path.join(path, folder), models):
        print("Looking file in '{}'.".format(model))
        for onnx in glob(os.path.join(model, "*.onnx")):
            tensor = os.path.split(onnx.replace(".onnx", ".trt"))[-1]
            if tensor in os.listdir(model):
                print("TensorRT file '{}' already exists.".format(tensor))
            else:
                print("Try converting '{}' to '{}'.".format(
                    tensor.replace(".trt", ".onnx"), tensor
                ))
                onnx2trt(onnx, "YOLO" in onnx)


main(("HiT", "YOLO"))
