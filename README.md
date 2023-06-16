# human_onnx_deploy
## Prerequisites 先决条件
需要先在Ubuntu 20.04 LTS安装
* CUDA 11.3
* OpenCV 4.5.2
* ONNX Runtime(CUDA) 1.8.0
## ONNX Models ONNX模型
[Dowload link 下载链接](https://www.jianguoyun.com/p/DfJ_d3AQn9HcCxjtzowFIAA)

其中各文件夹中包含对应的ONNX模型。

Each of these folders contains the corresponding ONNX model. 

## Human Matting 人像抠图
```bash
cd HumanMatting
g++ MODNet.cpp -o MODNet -lonnxruntime `pkg-config --libs opencv4`
./MODNet
```

## Human Detection 人物检测
```bash
cd HumanDetection
g++ RTMDet.cpp -o RTMDet -lonnxruntime `pkg-config --libs opencv4`
./RTMDet
```

## Person Pose Estimation 人体姿态估计
```bash
cd PersonPoseEstimation
g++ RTMPose.cpp -o RTMPose -lonnxruntime `pkg-config --libs opencv4`
./RTMPose
```
