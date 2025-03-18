# 代码结构

```yaml
Workspace
├── docs                            # 文档文件夹
├── images                          # 图片文件夹
├── scripts                         # 脚本文件夹
    ├── build.sh                    # 编译脚本
    └── onnx2trt.py                 # TensorRT 导出脚本
├── src                             # 源代码文件夹
    ├── client                      # 客户端代码
        ├── config                  # 存放配置文件
            └── names.txt           # 所有类别的名字
        ├── launch                  # 存放启动文件
            ├── camera.launch       # 启动相机客户端
            ├── image.launch        # 启动图片客户端
            ├── two_video.launch    # 启动两路视频客户端
            └── video.launch        # 启动视频客户端
        ├── scripts                 # 存放 Python 代码
            └── window.py           # 前端界面代码
        ├── src                     # 存放 C++ 代码（暂时为空）
        ├── CMakeLits.txt           # CMake 文件
        └── package.xml             # 代码包说明文件
    ├── server                      # 服务端代码
        ├── configs                 # 存放配置文件
            ├── mot.yaml            # MOT 算法超参数
            └── sot.yaml            # SOT 算法超参数 
        ├── include                 # 存放头文件
            ├── mot                 # 存放 MOT 头文件
                ├── byte_track.zip  # BtyeTrack 代码，编译后自动解压
                ├── ByteTrack.hpp   # ByteTrack 头文件
                ├── object.hpp      # 定义 MOT 和 Detection 的目标框
                └── YOLO.hpp        # YOLO 目标检测头文件
            ├── sot                 # 存放 SOT 头文件
                └── HiT.hpp         # HiT 跟踪器头文件
            └── logger              # TensorRT 推理头文件
        ├── launch                  # 存放启动文件
            ├── server.launch       # 启动服务端
            └── two_server.launch   # 启动两个服务端
        ├── models                  # 存放模型文件
            ├── EdgeSAM             # 存放 EdgeSAM 模型
                ├── decoder.onnx    # EdgeSAM 解码器
                └── encoder.onnx    # EdgeSAM 编码器
            ├── HiT                 # 存放 HiT 跟踪模型
                └── HiT.onnx        # HiT 模型
            ├── YOLO                # 存放 YOLO 目标检测模型
                └── YOLOv6s.onnx    # YOLOv6s_mbla 模型
        ├── scripts                 # 存放 Python 代码
            └── point2box.py        # EdgeSAM 服务端代码
        ├── src                     # 存放 C++ 代码
            └── server.cpp          # 服务端代码
        ├── CMakeLits.txt           # CMake 文件
        └── package.xml             # 代码包说明文件
    └── tracking_msgs               # 通信接口代码
        ├── src                     # 存放 C++ 代码（暂时为空）
        ├── srv                     # 存放服务请求协议
            ├── Image.srv           # 图像请求协议
            ├── ImageWithBoxes.srv  # 图像和检测框请求协议
            └── Point2Box.srv       # SAM 服务请求协议
        ├── CMakeLits.txt           # CMake 文件
        └── package.xml             # 代码包说明文件
├── .catkin_workspace               # 工作空间文件
└── README.md                       # 使用说明文档
```