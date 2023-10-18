#ifndef SRCH_H_ 
#define SRCH_H_

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "NetOpenVino.h"


//��������� ��� ���������� �������� 
constexpr auto alpha = 0.9;


/*
����� ��� ������ ������� 
� ������� ������� ������ NeuralNetworkDetector � ��������� ������� onnx  
Input
detectorOnnx - ������� ������ NeuralNetworkDetector (��� ������� ���������), image - ����������� �� ���� ����, vectorAll - ������ �� ����� ������������ ���������, idGlobal - id �������
Output 
����������� ������ � ���������� ����������� 
*/
void SearchVector(NeuralNetworkDetector detectorOnnx, const cv::Mat& image, std::vector<std::vector<float>>& vectorAll, const int& idGlobal);


/*
����� ��� ��������� ��������
� ������� ������� ������ NeuralNetworkDetector � ��������� ������� onnx
Input
vector1 - ����������� �������, vector2 - ������ �������� ������� 
Output
������������� �������������� 
*/
auto SearchCompare(const std::vector<float>& vector1, const std::vector<float>& vector2);



/*
����� ��� �������� ������ ��������
� ������� ������� ������ NeuralNetworkDetector � ��������� ������� onnx
Input
detectorOnnx - ������� ������ NeuralNetworkDetector (��� ������� ���������), image - ����������� �� ���� ����, vector - ������ �� ����� ������������ ���������, idGlobal - id �������, param - �������� ���������, arrID - ������ ��� ���������� ���������� ���������� �������� 
Output
������������ arrID
*/
void CountCompare(NeuralNetworkDetector detectorOnnx, const cv::Mat& image, std::vector<std::vector<float>>& vector, const int& idGlobal, const float& param, std::vector<int>& arrID);


#endif
