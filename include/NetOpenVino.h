#ifndef NETOPV_H_ 
#define NETOPV_H_
#include <cstdio>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>


class NeuralNetworkDetector {
public:
    NeuralNetworkDetector() {};


    /*
    метод для загрузки сети и получения информации о входе и выходе сети 
    Input
    modelFormat - формат модели,  modelPath - путь до модели сети, device - устройство CPU или GPU,  size - размер входа сети
    Output
    модель сети  
    */
    void ReadNet(const std::string& modelFormat, const std::string& modelPath, const std::string& device, const cv::Size& size);
  

    /*
    метод для подготовки входных и выходных блобов
    Input
    modelFormat - формат модели,  modelPath - путь до модели сети, device - устройство CPU или GPU,  size - размер входа сети
    Output
    модель сети
    */
    void getInputOutputInfo();


    /*
    метод для получения выходных данных сети 
    Input
    Frame - входное изображение
    Output
    detection - выходные данные сети
    */
    const float* Forward(const cv::Mat& Frame);


    /*
    метод для преобразования выходных данных сети openvino
    Input
    detection выходные данные сети,  detections - для записи результата,  image - изображение для получения размера, size - минимальный размер бокса, sizeA - максимальный размер бокса
    Output
    detections - выходные данные сети в формате вектора 4 точек 
    */
    void outputOpenVino(const float* detection, std::vector<cv::Rect>& detections, const cv::Mat& image, int size, int sizeA);
  

private:
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest inferRequest;
    InferenceEngine::OutputsDataMap outputs_info;
    InferenceEngine::InputsDataMap inputs_info;
    InferenceEngine::DataPtr output_info;
    std::string output_name;
    InferenceEngine::InputInfo::Ptr input_info;
    std::string input_name;
    size_t max_proposal_count;
    size_t object_size;

    std::string deviceNet;
    size_t batch_size;

    cv::Size sizeNet;
};

#endif