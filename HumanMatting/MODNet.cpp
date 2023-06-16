#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <core/session/onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;


void HWCToCHW(const cv::Mat& src, float *dst) {
    std::vector<Mat> vec;
    cv::split(src, vec);
    int hw = src.rows * src.cols;
    memcpy(dst + hw * 0, vec[0].data, hw * sizeof(float));
    memcpy(dst + hw * 1, vec[1].data, hw * sizeof(float));
    memcpy(dst + hw * 2, vec[2].data, hw * sizeof(float));
}

int main(){
    Mat img = imread("./demo.jpg", 1);
    Rect top_left_roi(107, 0, img.rows, img.rows);
    Mat img_crop = img(top_left_roi);
    Mat resize_img;
    resize(img_crop, resize_img, Size(512, 512), INTER_AREA);
    Mat blob0 = resize_img.clone();
    blob0.convertTo(blob0, CV_32FC3);
    Mat blob1 = (blob0 - 127.5) / 127.5;
    float blob[1 * 3 * 512 * 512];
    HWCToCHW(blob1, blob);
    

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

    OrtCUDAProviderOptions cuda_options{
          0,
          OrtCudnnConvAlgoSearch::EXHAUSTIVE,
          std::numeric_limits<size_t>::max(),
          0,
          true
      };

    session_options.AppendExecutionProvider_CUDA(cuda_options);
    const char* model_path = "./modnet.onnx";
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    vector<const char*> input_node_names = { "input" };
    vector<const char*> output_node_names = { "output" };

    std::vector<int64_t> input_node_dims = {1, 3, 512, 512};
    size_t input_tensor_size = 1 * 3 * 512 * 512;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
    std::vector<int64_t> inputShape = input_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // cout << "Input Dims: " << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << endl;

    auto output_tensors = session.Run(Ort::RunOptions(nullptr), input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_node_names.size());
    std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // cout << "Output Dims: " << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << endl;
    
    auto* rawOutput = output_tensors[0].GetTensorData<float>();
    Mat output(512, 512, CV_32FC1, (float*)rawOutput);
    output = output * 255;
    output.convertTo(output, CV_8UC1);
    imwrite("output.jpg", output);

    return 0;
}

