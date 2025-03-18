## 1. 环境配置

本工程代码运行在 `NVIDIA JetSon Orin NX` 上，请参照[这里](env.md)准备软硬件环境。

## 2. 代码结构

如果你想了解本工程的代码结构，请点击[这里](structure.md)查看。

## 3. 使用说明

以下所有命令都在本工程根目录下执行。

### 3.1 首次运行

第一次运行本工程代码时，需要先进行模型导出、编译和环境变量的设置。

#### 3.1.1 导出模型

使用以下命令一键导出模型：

```shell
python3 scripts/onnx2trt.py
```

导出模型需要数十分钟时间，请耐心等待。

#### 3.1.2 编译源码

使用以下命令一键编译：

```shell
bash scripts/build.sh
```

#### 3.1.3 设置环境变量（可选）

使用以下命令打开 `.bashrc` 文件：

```shell
sudo gedit ~/.bashrc
```

在打开的文件的最后一行添加以下代码：

```shell
source ${PATH_TO_PROJECT}/devel/setup.bash
```

请将 `${PATH_TO_PROJECT}` 更换为本工程的路径，请使用绝对路径。

添加完成后，保存退出即可。

### 3.2 启动服务端

打开一个终端，输入以下命令启动服务端：

```shell
roslaunch server server.launch
```

### 3.3 启动客户端

#### 3.3.1 视频客户端

打开一个终端，使用以下命令启动视频客户端界面：

```shell
roslaunch client video.launch video:=${PATH_TO_VIDEO}
```

请将 `${PATH_TO_VIDEO}` 替换为你的视频路径，建议使用绝对路径。

如果需要保存视频和单目标跟踪的 `txt` 结果文件，需要加上 `save:=true`

```shell
roslaunch client video.launch video:=${PATH_TO_VIDEO} save:=true
```

视频和结果会保存在 `logs` 文件夹下，`txt` 文件中每一行的四个数表示目标的坐标（左上角xy，右下角xy）

在其他模式下（相机和图片），加上 `save:=true` 同样可以实现保存功能，后续不再赘述

#### 3.3.2 相机客户端

打开一个终端，使用以下命令启动相机客户端界面：

```shell
roslaunch client camera.launch
```

默认打开编号为 `0` 的摄像头，如果想使用其他摄像头，可以改用以下指令：
```shell
roslaunch client camera_demo.launch camera_id:=${CAMERA_ID}
```

请将 `${CAMERA_ID}` 替换为你想使用的相机的编号。

#### 3.3.3 图片客户端

打开一个终端，使用以下命令启动图片客户端界面：

```shell
roslaunch client image.launch image:=${PATH_TO_IMAGES}
```

请将 `${PATH_TO_IMAGES}` 替换为你的图片文件夹路径，建议使用绝对路径，图片文件夹中的图片按照以下规则命名：

```yaml
${PATH_TO_IMAGES}
├── 000001.jpg
├── 000002.jpg
├── 000003.jpg
└── ······
```

### 3.4 修改超参数

算法的超参数在 `src/server/configs` 中，`sot.yaml` 定义了 `SOT` 的超参数，`mot.yaml` 定义了 `MOT` 和 `DET` 的超参数。

修改超参数后，重新启动服务端即可加载超参数，无需重新编译。

#### 3.4.1 SOT 的超参数

```yaml
# 神经网络后处理相关
margin: 10.0           # 目标框到图像边缘的最大距离
mlp_threshold: 0.6     # 用于判断是否丢失的阈值，神经网络输出小于该值则判定丢失，范围：[0, 1]
score_threshold: 0.35  # 用于判定是否跟踪成功的阈值，神经网络输出小于该值则判定跟踪失败，范围：[0, 1]

# 运动预测相关
beta: 0.001          # 目标丢失后，竖直方向搜索区域的扩张系数
alpha: 0.002         # 目标丢失后，水平方向搜索区域的扩张系数
momentum: 100        # 表示运动预测依赖目标最近多少次历史运动的信息
area_threshold: 1.5  # 重找回的目标框的面积和模板的面积比例阈值，超过这个阈值不会找回，必须大于等于1

# 全局重找回相关
retrieve_wait_time: 1.0  # 目标丢失后，等待多少秒开始全局重找回
retrieve_threshold: 0.8  # 全局重找回的阈值，重找回目标和模板的相似度大于该阈值才会找回
use_global_retrieval: 1  # 是否使用全局重找回，0 表示不使用，1 表示仅 SOT 模式下使用，2 表示总是使用
```

#### 3.4.2 MOT 和 DET 的超参数

```yaml
# 神经网络相关
num_classes: 7     # 神经网络输出的类别的数量
infer_size: 480    # 神经网络输入的图像的尺寸
outputs_dim: 4725  # 神经网络输入的向量的维度

# 阈值，所有阈值的范围都是 [0, 1]
nms_threshold: 0.45    # NMS 阈值
match_threshold: 0.5   # ByteTrack 的匹配阈值，小于该值不匹配
detect_threshold: 0.5  # YOLO 的检测阈值，网络输出大于该值就认为有目标
```

### 3.5 指定模型路径

在默认情况下，`HiT` 模型将从 `src/server/models/HiT/HiT.trt` 中加载，`YOLO` 模型将从 `src/server/models/YOLO/YOLOv6s.trt` 中加载。在启动服务端时，可以使用以下命令更改模型加载路径：

```
roslaunch server server.launch HiT:=${PATH_TO_HIT} YOLO:=${PATH_TO_YOLO}
```

其中，`YOLO` 模型支持加载 `YOLOv5` 和 `YOLOv6`。加载前，请修改 `src/server/configs/mot.yaml` 的前三个超参数，以匹配模型的输出。

### 3.6 运行两路视频

我们提供了两个 `launch` 文件，直接运行这两个 `launch` 文件可实现两路视频的同时处理。

打开一个终端，使用以下命令启动服务端：

```shell
roslaunch server two_server.launch
```

再打开一个终端，使用以下命令启动视频客户端界面：

```shell
roslaunch client two_video.launch video1:=${PATH_TO_VIDEO1} video2:=${PATH_TO_VIDEO2}
```

注意：请将 `${PATH_TO_VIDEO1}` 和 `${PATH_TO_VIDEO2}` 替换为你的视频路径，建议使用绝对路径。

事实上，`two_server.launch` 只是将 `two_server.launch` 中的内容写了两遍，并修改了服务的名字而已。
你可以按照以下教程编写 $n$ 个服务端 $n$ 路视频的 `launch` 文件。

以 $2$ 个服务端处理 $2$ 路视频为例，服务端和客户端的 `launch` 文件编写如下：

#### 3.6.1 服务端

```xml
<launch>
    <!-- 模型路径，有几个服务端就写几次，这里以 2 个为例 -->
    <arg name="HiT1" default="$(find server)/models/HiT/HiT.trt" />
    <arg name="HiT2" default="$(find server)/models/HiT/HiT.trt" />
    <arg name="YOLO1" default="$(find server)/models/YOLO/YOLOv6s.trt" />
    <arg name="YOLO2" default="$(find server)/models/YOLO/YOLOv6s.trt" />

    <!-- 加载超参数，无需更改 -->
    <rosparam file="$(find server)/configs/sot.yaml" command="load" />
    <rosparam file="$(find server)/configs/mot.yaml" command="load" />

    <!-- 启动跟踪服务，有几个服务端就写几次，这里以 2 个为例 -->
    <node name="server1" pkg="server" type="server" output="screen" 
          args="$(arg HiT1) $(arg YOLO1)" >
        <param name="save" value="false" type="bool" />
        <param name="mot_server" value="/MOT/frame1" type="string" />
        <param name="sot_server" value="/SOT/frame1" type="string" />
        <param name="det_server" value="/DET/frame1" type="string" />
        <param name="box_server" value="/SOT/boxes1" type="string" />
        <param name="reset_server" value="/MOT/reset1" type="string" />
        <param name="sot_mot_server" value="/MOT/sot1" type="string" />
        <param name="sot_det_server" value="/DET/sot1" type="string" />
    </node>
    <node name="server2" pkg="server" type="server" output="screen" 
          args="$(arg HiT2) $(arg YOLO2)" >
        <param name="save" value="false" type="bool" />
        <param name="mot_server" value="/MOT/frame2" type="string" />
        <param name="sot_server" value="/SOT/frame2" type="string" />
        <param name="det_server" value="/DET/frame2" type="string" />
        <param name="box_server" value="/SOT/boxes2" type="string" />
        <param name="reset_server" value="/MOT/reset2" type="string" />
        <param name="sot_mot_server" value="/MOT/sot2" type="string" />
        <param name="sot_det_server" value="/DET/sot2" type="string" />
    </node>

    <!-- EdgeSAM 服务，无需更改 -->
    <node name="point2box" pkg="server" type="point2box.py" output="screen" />
</launch>
```

#### 3.6.2 客户端

```xml
<launch>
    <!-- 视频路径，有几路视频就写几个，这里以 2 个为例 -->
    <arg name="video1" />
    <arg name="video2" />

    <!-- 启动客户端，有几路视频就写几个，这里以 2 个为例 -->
    <node name="client1" pkg="client" type="window.py" output="screen"
          args="video $(arg video1) Window1" >
        <param name="mot1" value="/MOT/sot1" type="string" />
        <param name="det1" value="/DET/sot1" type="string" />
        <param name="mot" value="/MOT/frame1" type="string" />
        <param name="sot" value="/SOT/frame1" type="string" />
        <param name="det" value="/DET/frame1" type="string" />
        <param name="box" value="/SOT/boxes1" type="string" />
        <param name="res" value="/MOT/reset1" type="string" />
    </node>
    <node name="client2" pkg="client" type="window.py" output="screen"
          args="video $(arg video2) Window2" >
        <param name="mot1" value="/MOT/sot2" type="string" />
        <param name="det1" value="/DET/sot2" type="string" />
        <param name="mot" value="/MOT/frame2" type="string" />
        <param name="sot" value="/SOT/frame2" type="string" />
        <param name="det" value="/DET/frame2" type="string" />
        <param name="box" value="/SOT/boxes2" type="string" />
        <param name="res" value="/MOT/reset2" type="string" />
    </node>
</launch>
```
