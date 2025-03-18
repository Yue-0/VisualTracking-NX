# 在 JetSon Orin NX 上部署轻量化视觉跟踪模型

## 简介

![demo](images/SOT.gif)

本工程实现了 HiT 模型的轻量化，并增加了遮挡判断和位置预测的功能，[查看模型结构](images/HiT.png)。

本工程将训练好的 __SAM__、__单目标跟踪__、__多目标跟踪__ 和 __目标检测__ 模型集成部署在 JetSon Orin NX 边缘计算设备上，可以实现：
1. 基于 [EdgeSAM](https://arxiv.org/abs/2312.06660) 的 Point2Box 
2. 基于轻量化 [HiT](https://arxiv.org/abs/2308.06904) 的单目标跟踪
3. 基于 [ByteTrack](https://arxiv.org/abs/2110.06864) 的多目标跟踪
4. 基于 [YOLOv6](https://arxiv.org/abs/2209.02976) 的目标检测

点击[这里](docs/README.md)查看使用文档。

注：本工程仅实现训练好的模型在 JetSon Orin NX 上的推理功能，不包含训练代码。

## 致谢

* HiT 的训练源代码和预训练模型由 [Kang Ben](https://github.com/kangben258) 和 [Chen Xin](https://github.com/chenxin-dlut) 提供。

* ByteTrack 的 C++ 代码实现大量参考了 [hpc203](https://github.com/hpc203) 的 [开源仓库](https://github.com/hpc203/bytetrack-opencv-onnxruntime)。

* YOLOv6 的预训练模型和微调使用 [美团的开源代码](https://github.com/meituan/yolov6)。

* EdgeSAM 模型来自 [Zhou Chong](https://github.com/chongzhou96) 的 [开源仓库](https://github.com/chongzhou96/EdgeSAM)。
