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

typedef struct Bbox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
}Bbox;

/*  计算iou  */
float iou(Bbox box1,Bbox box2)
{
    float insection_x1 = max(box1.x1,box2.x1);
    float insection_y1 = max(box1.y1,box2.y1);
    float insection_x2 = min(box1.x2,box2.x2);
    float insection_y2 = min(box1.y2,box2.y2);
    if(insection_x1>=insection_x2 || insection_y1>=insection_y2)
        return 0;
    float insection_area = (insection_x2 - insection_x1) * (insection_y2 - insection_y1);
    float union_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1) + (box2.x2 - box2.x1) * (box2.y2 - box2.y1) - insection_area;
    float iou = insection_area / union_area;
    return iou;
}

/* 将bbx按照confidence从高到低排序 */
bool sort_score(Bbox box1,Bbox box2)
{
    return (box1.score > box2.score);
}

// 计算nms
vector<Bbox> nms(vector<Bbox>vec_boxs, float threshold)
{
    vector<Bbox>  res;
    while(vec_boxs.size() > 0)
    {
        sort(vec_boxs.begin(), vec_boxs.end(), sort_score); // 将bbx按照confidence从高到低排序
        res.push_back(vec_boxs[0]);
        for(int i =0;i<vec_boxs.size()-1;i++)
        {
            float iou_value=iou(vec_boxs[0],vec_boxs[i+1]);
            if (iou_value>threshold)
            {
                vec_boxs.erase(vec_boxs.begin()+i+1);
            }
        }
        vec_boxs.erase(vec_boxs.begin());  // res 已经保存，所以可以将最大的删除了
    }
    return res;
}

int main()
{
    clock_t start, end;
    // input image read
    Mat img = imread("./demo.jpg", 1);
    // crop image to same height and width
    cout << "(width, height)"<< img.size() << endl;
    cout << "Width : " << img.cols << endl;
    cout << "Height: " << img.rows << endl;
    copyMakeBorder(img,img,107,108,0,0, cv::BORDER_CONSTANT,0);
    
    Mat blob0 = img.clone();
    blob0.convertTo(blob0, CV_32FC3);
    Mat blob1 = blob0 / 255.0;
    Mat rgbchannel[3];
    split(blob1, rgbchannel);
    rgbchannel[0] = (rgbchannel[0] - 0.406) / 0.225;
    rgbchannel[1] = (rgbchannel[1] - 0.456) / 0.224;
    rgbchannel[2] = (rgbchannel[2] - 0.485) / 0.229;
    Mat blob2;
    merge(rgbchannel, 3, blob2);
    float blob[1 * 3 * 640 * 640];
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
    

    const char* model_path = "./RTMDet-m_640x640.onnx";
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    vector<const char*> input_node_names = { "input" };
    vector<const char*> output_node_names = { "dets", "labels" };
    // cout << "Input Name: " << input_name << endl;
    // cout << "Output Name: " << output_name << endl;
    // cout << "Output Dims: " << output_dims[0] << "," << output_dims[1] << "," << output_dims[2] << "," << output_dims[3] << endl;

    
    std::vector<int64_t> input_node_dims = {1, 3, 640, 640};
    size_t input_tensor_size = 1 * 3 * 640 * 640;

    // cout << blob[0] << "," << blob[1] << "," << blob[2] << endl;
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
    std::vector<int64_t> inputShape = input_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    cout << "Input Dims: " << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << endl;
    
    auto output_tensors = session.Run(Ort::RunOptions(nullptr), input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_node_names.size());

    auto* raw_dets = output_tensors[0].GetTensorData<float>();
    auto* raw_labels = output_tensors[1].GetTensorData<int>();
    size_t count_dets = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    size_t count_labels = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
    vector<float> dets(raw_dets, raw_dets + count_dets);
    vector<float> labels(raw_labels, raw_labels + count_labels);
    size_t chn_cnt_dets = 5;
    size_t cnt_dets = count_dets / chn_cnt_dets;
    vector<Bbox> bbox_dets;
    for(int index=0;index<cnt_dets;index++){
        vector<float> this_channel_dets;
        this_channel_dets.assign(dets.begin() + index * chn_cnt_dets, dets.begin() + (index + 1) * chn_cnt_dets);
        if(this_channel_dets[4]>0.5){
            Bbox this_dets = {this_channel_dets[0], this_channel_dets[1], this_channel_dets[2], this_channel_dets[3], this_channel_dets[4]};
            bbox_dets.push_back(this_dets);
        }
        // rectangle(img, cv::Point(int(this_channel_dets[0]), int(this_channel_dets[1])), cv::Point(int(this_channel_dets[2]), int(this_channel_dets[3])), cv::Scalar(0, 255, 0),2);
    }
    vector<Bbox> bbox_nms = nms(bbox_dets, 0.7);
    int bbox_cnt = bbox_nms.size();
    for(int index=0;index<bbox_cnt;index++){
        rectangle(img, cv::Point(int(bbox_nms[index].x1), int(bbox_nms[index].y1)), cv::Point(int(bbox_nms[index].x2), int(bbox_nms[index].y2)), cv::Scalar(0, 255, 0),1);
    }
    imwrite("output.jpg", img);

    return 0;
}