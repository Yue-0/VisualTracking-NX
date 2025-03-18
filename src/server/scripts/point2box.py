#!/usr/bin/env python3

## Import standard libraries
from os import path
from time import time as now

## Import third-party libraries
import cv2 as cv
import rospy as rp
import numpy as np
from cv_bridge import CvBridge
from onnxruntime import InferenceSession

## Import custom service
import tracking_msgs.srv as srv

__author__ = "YueLin"


class EdgeSAM:
    """EdgeSAM using ONNX inference"""
    def __init__(self, encoder: str, decoder: str):
        self.feature = None
        self.encoder = InferenceSession(
            encoder, None, ["CUDAExecutionProvider"]
        )
        self.decoder = InferenceSession(
            decoder, None, ["CUDAExecutionProvider"]
        )
        self.std = np.array([58.395, 57.12, 57.375])[
            np.newaxis, :, np.newaxis, np.newaxis
        ]
        self.mean = np.array([123.675, 116.28, 103.53])[
            np.newaxis, :, np.newaxis, np.newaxis
        ]
        self.size = [0x400, None, None]
    
    def encode(self, image: np.ndarray) -> None:
        """EdgeSAM encode"""
        img = self.resize(image)
        self.size[1] = image.shape[:2]
        self.size[2] = tuple(img.shape[-2:])
        self.feature = self.encoder.run(None, {
            "image": self.preprocess(img).astype(np.float32)
        })[0]

    def decode(self, label: np.ndarray, point: np.ndarray) -> np.ndarray:
        """EdgeSAM decode"""
        point = self.resize(point, self.size[1])
        return self.post_process(self.decoder.run(None, {
            "image_embeddings": self.feature,
            "point_coords": point.astype(np.float32),
            "point_labels": label.astype(np.float32)
        })[1]) > 0
    
    def resize(self, data: np.ndarray, size: tuple = None) -> np.ndarray:
        """Resize image (size is None) or coordinate (size is not None)"""
        if size is not None:
            h, w = size
        else:
            h, w = data.shape[:2]
        scale = self.size[0] / max(h, w)
        h, w = int(h * scale + 0.5), int(w * scale + 0.5)
        if size is not None:
            result = data.copy()
            result[..., 0] *= w / size[1]
            result[..., 1] *= h / size[0]
            return result
        return cv.resize(data, (w, h)).transpose(2, 0, 1)[np.newaxis]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """EdgeSAM pre-processing, normalization and padding"""
        h, w = image.shape[-2:]
        return np.pad((image - self.mean) / self.std, (
            (0, 0), (0, 0), (0, self.size[0] - h), (0, self.size[0] - w)
        ), "constant", constant_values=0)
    
    def post_process(self, mask: np.ndarray) -> np.ndarray:
        """EdgeSAM post-processing, resize mask"""
        return cv.resize(
            cv.resize(
                mask.squeeze(0).transpose(1, 2, 0), (self.size[0],) * 2
            )[:self.size[2][0], :self.size[2][1], :], 
            (self.size[1][1], self.size[1][0])
        ).transpose(2, 0, 1)[np.newaxis]


class Point2Box:
    """Point to box service"""
    def __init__(self, encoder: str, decoder: str):
        rp.loginfo("[SAM] Loading EdgeSAM model from {}.".format(
            path.split(encoder)[0]
        ))
        self.label = np.zeros((1, 0))
        self.point = np.zeros((1, 0, 2))
        self.model = EdgeSAM(encoder, decoder)
        rp.loginfo("[SAM] Warming up. This would take a while...")
        for _ in range(2):
            self.model.encode(np.random.randint(
                0x100, size=(480, 640, 3), dtype=np.uint8
            ))
            self.model.decode(
                np.concatenate((self.label, [[1]]), axis=1),
                np.concatenate((self.point, [[[320, 240]]]), axis=1)
            )
        self.msg2img = CvBridge().imgmsg_to_cv2
        self.encoder = rp.Service(
            "/SOT/frame0", srv.Image, self.encode
        )
        self.decoder = rp.Service(
            "/SOT/point2box", srv.Point2Box, self.point2box
        )
        rp.loginfo("[SAM] Service is ready.")

    def encode(self, image: srv.Image) -> srv.ImageResponse:
        rp.loginfo("[SAM] Model encoding...")
        self.model.encode(cv.cvtColor(
            self.msg2img(image.image, "bgr8"),
            cv.COLOR_BGR2RGB
        ))
        rp.loginfo("[SAM] Reay to segmention.")
        return srv.ImageResponse()
    
    def point2box(self, point: srv.Point2BoxRequest) -> srv.Point2BoxResponse:
        rp.loginfo("[SAM] Running point2box, coordinate: ({}, {})".format(
            point.x, point.y
        ))
        time = now()
        mask = np.argwhere(self.model.decode(
            np.concatenate((self.label, [[1]]), axis=1),
            np.concatenate((self.point, [[[point.x, point.y]]]), axis=1)
        )[0][0] != 0)
        y1, x1 = mask.min(axis=0)
        y2, x2 = mask.max(axis=0)
        time = now() - time
        rp.loginfo("[SAM] Segment successful, took %.3fs." % time)
        return srv.Point2BoxResponse(*map(int, (x1, x2, y1, y2)), time)


if __name__ == "__main__":

    ## Initialize service node
    rp.init_node("point2box")
    
    ## Path to ONNX file
    onnx = path.join(
        path.split(path.split(__file__)[0])[0],
        "models", "EdgeSAM", "{}coder.onnx"
    )

    ## Initialize server
    _ = Point2Box(onnx.format("en"), onnx.format("de"))
    
    ## Start service
    rp.spin()
