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

tuple<size_t, float> argmax(vector<float> v){
    float maxVal = -10000.0;
    size_t maxIndex = 0;
    int length = v.size();
    for(int i=0;i<length;i++){
      if(v[i]>maxVal){
        maxVal = v[i];
        maxIndex = i;
      }
    }
    return tuple<size_t, float>(maxIndex, maxVal);
}

void HWCToCHW(const cv::Mat& src, float *dst) {
    std::vector<Mat> vec;
    cv::split(src, vec);
    int hw = src.rows * src.cols;
    memcpy(dst + hw * 0, vec[0].data, hw * sizeof(float));
    memcpy(dst + hw * 1, vec[1].data, hw * sizeof(float));
    memcpy(dst + hw * 2, vec[2].data, hw * sizeof(float));
}


int main()
{
    // input image read
    Mat img = imread("./demo.jpg", 1);
    // crop image to same height and width
    cout << "(width, height)"<< img.size() << endl;
    cout << "Width : " << img.cols << endl;
    cout << "Height: " << img.rows << endl;
    Rect top_left_roi(160+60, 0, 319, img.rows);
    Mat img_crop = img(top_left_roi);
    // cout << "(width, height)"<< img_crop.size() << endl;
    Mat resize_img;
    resize(img_crop, resize_img, Size(288, 384), INTER_LINEAR);
    // cout << "(width, height)"<< resize_img.size() << endl;
    Mat blob0 = resize_img.clone();
    blob0.convertTo(blob0, CV_32FC3);
    Mat blob1 = blob0 / 255.0;
    Mat rgbchannel[3];
    split(blob1, rgbchannel);
    rgbchannel[0] = (rgbchannel[0] - 0.406) / 0.225;
    rgbchannel[1] = (rgbchannel[1] - 0.456) / 0.224;
    rgbchannel[2] = (rgbchannel[2] - 0.485) / 0.229;
    Mat blob2;
    merge(rgbchannel, 3, blob2);
    float blob[1 * 3 * 288 * 384];
    HWCToCHW(blob2, blob);

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

    const char* model_path = "./RTMPose-l_384_288.onnx";
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    vector<const char*> input_node_names = { "input" };
    vector<const char*> output_node_names = { "simcc_x", "simcc_y" };
    // cout << "Input Name: " << input_name << endl;
    // cout << "Output Name: " << output_name << endl;
    // cout << "Output Dims: " << output_dims[0] << "," << output_dims[1] << "," << output_dims[2] << "," << output_dims[3] << endl;

    
    std::vector<int64_t> input_node_dims = {1, 3, 384, 288};
    size_t input_tensor_size = 1 * 3 * 384 * 288;

    // cout << blob[0] << "," << blob[1] << "," << blob[2] << endl;
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
    std::vector<int64_t> inputShape = input_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    cout << "Input Dims: " << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << endl;

    auto output_tensors = session.Run(Ort::RunOptions(nullptr), input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_node_names.size());
    // std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // cout << "Output Dims: " << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << endl;
    // cout << "time = " << double(end-start)/CLOCKS_PER_SEC << endl;
    
    // auto* rawInput = input_tensors[0].GetTensorData<float>();
    // cout << rawInput[0] << "," << rawInput[1] << "," << rawInput[2] << endl;

    auto* rawOutputX = output_tensors[0].GetTensorData<float>();
    auto* rawOutputY = output_tensors[1].GetTensorData<float>();
    size_t countX = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    size_t countY = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
    vector<float> outputX(rawOutputX, rawOutputX + countX);
    vector<float> outputY(rawOutputY, rawOutputY + countY);
    // cout << outputX[0] << "," << outputX[1] << "," << outputX[2] << endl;
    // cout << outputY[0] << "," << outputY[1] << "," << outputY[2] << endl;
    // cout << rawOutputX[0] << "," << rawOutputX[1] << "," << rawOutputX[2] << endl;
    size_t chn_cnt_x = 576;
    size_t chn_cnt_y = 768;
    for(int index=0;index<133;index++){
      vector<float> this_channel_x, this_channel_y;
      this_channel_x.assign(outputX.begin() + index * chn_cnt_x, outputX.begin() + (index + 1) * chn_cnt_x);
      // cout << this_channel.size() << endl;
      tuple<size_t, float> x_pack = argmax(this_channel_x);
      size_t x = get<0>(x_pack);
      float xval = get<1>(x_pack);
      this_channel_y.assign(outputY.begin() + index * chn_cnt_y, outputY.begin() + (index + 1) * chn_cnt_y);
      // cout << this_channel.size() << endl;
      tuple<size_t, float> y_pack = argmax(this_channel_y);
      size_t y = get<0>(y_pack);
      float yval = get<1>(y_pack);
      float confidence = xval * yval;
      cout << confidence << endl;
      if(confidence>=0.5){
          circle(resize_img, Point(int(x/2), int(y/2)), 1, Scalar(255, 0, 0), FILLED);
      }
    }

    imwrite("output.jpg", resize_img);
    
    return 0;
}