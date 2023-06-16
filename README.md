# human_onnx_deploy
## prerequest
OpenCV 4.5.2
ONNX Runtime(CUDA) 1.8.0
## human matting
cd HumanMatting
g++ MODNet.cpp -o MODNet -lonnxruntime `pkg-config --libs opencv4`
./MODNet

## human detection
cd HumanDetection
g++ RTMDet.cpp -o RTMDet -lonnxruntime `pkg-config --libs opencv4`
./RTMDet

## person pose estimation
cd PersonPoseEstimation
g++ RTMPose.cpp -o RTMPose -lonnxruntime `pkg-config --libs opencv4`
./RTMPose
