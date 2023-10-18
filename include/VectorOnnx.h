#ifndef SRCH_H_ 
#define SRCH_H_

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "NetOpenVino.h"


//константа для накопления векторов 
constexpr auto alpha = 0.9;


/*
метод для поиска вектора 
с помощью объекта класса NeuralNetworkDetector и нейросети формата onnx  
Input
detectorOnnx - объекта класса NeuralNetworkDetector (для запуска нейросети), image - изображение на вход сети, vectorAll - вектор со всеми накопленными векторами, idGlobal - id объекта
Output 
Накопленный вектор с нескольких изображений 
*/
void SearchVector(NeuralNetworkDetector detectorOnnx, const cv::Mat& image, std::vector<std::vector<float>>& vectorAll, const int& idGlobal);


/*
метод для сравнения векторов
с помощью объекта класса NeuralNetworkDetector и нейросети формата onnx
Input
vector1 - накполенные векторы, vector2 - вектор текущего объекта 
Output
сравнительная характеристика 
*/
auto SearchCompare(const std::vector<float>& vector1, const std::vector<float>& vector2);



/*
метод для подсчета схожих векторов
с помощью объекта класса NeuralNetworkDetector и нейросети формата onnx
Input
detectorOnnx - объекта класса NeuralNetworkDetector (для запуска нейросети), image - изображение на вход сети, vector - вектор со всеми накопленными векторами, idGlobal - id объекта, param - параметр сравнения, arrID - массив для накполения количества подходящих векторов 
Output
заполененный arrID
*/
void CountCompare(NeuralNetworkDetector detectorOnnx, const cv::Mat& image, std::vector<std::vector<float>>& vector, const int& idGlobal, const float& param, std::vector<int>& arrID);


#endif
