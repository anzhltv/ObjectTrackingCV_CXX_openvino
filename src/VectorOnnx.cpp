#include "VectorOnnx.h"
#include "operator.h"


/*
метод для поиска вектора
с помощью объекта класса NeuralNetworkDetector и нейросети формата onnx
Input
detectorOnnx - объекта класса NeuralNetworkDetector (для запуска нейросети), image - изображение на вход сети, vectorAll - вектор со всеми накопленными векторами, idGlobal - id объекта
Output
Накопленный вектор с нескольких изображений
*/
void SearchVector(NeuralNetworkDetector detectorOnnx, const cv::Mat& image, std::vector<std::vector<float>>& vectorAll, const int& idGlobal)
{
	auto vector = detectorOnnx.Forward(image);
	std::vector<float> vectorOut(vector, vector + 256);
	if (vectorAll[idGlobal].empty())
		vectorAll[idGlobal] = vectorOut;
	else
		vectorAll[idGlobal] = vectorAll[idGlobal] * alpha + vectorOut * (1 - alpha);
}


/*
метод для сравнения векторов
с помощью объекта класса NeuralNetworkDetector и нейросети формата onnx
Input
vector1 - накполенные векторы, vector2 - вектор текущего объекта
Output
сравнительная характеристика
*/
auto SearchCompare(const std::vector<float>& vector1, const std::vector<float>& vector2)
{
	float mult = 0, sqrA = 0, sqrB = 0;
	for (int i = 0; i < vector1.size(); i++)
	{
		auto a = vector1[i];
		auto b = vector2[i];
		mult += a * b;
		sqrA += pow(a, 2);
		sqrB += pow(b, 2);
	}
	return mult / sqrt(sqrA * sqrB);
}


/*
метод для подсчета схожих векторов
с помощью объекта класса NeuralNetworkDetector и нейросети формата onnx
Input
detectorOnnx - объекта класса NeuralNetworkDetector (для запуска нейросети), image - изображение на вход сети, vector - вектор со всеми накопленными векторами, idGlobal - id объекта, param - параметр сравнения, arrID - массив для накполения количества подходящих векторов
Output
заполененный arrID
*/
void CountCompare(NeuralNetworkDetector detectorOnnx, const cv::Mat& image, std::vector<std::vector<float>>& vector, const int& idGlobal, const float& param, std::vector<int>& arrID)
{
	SearchVector(detectorOnnx, image, vector, idGlobal);
	auto max = 0, obj = -1;
	for (int i = 0; i < idGlobal; i++)
	{
		if (!vector[idGlobal].empty() && !vector[i].empty())
		{
			auto compare = SearchCompare(vector[idGlobal], vector[i]);
			if (compare > param)
			{
				if (max < compare)
				{
					max = compare;
					obj = i;
				}
			}
			if (obj >= 0)
				++arrID[obj];
		}
	}
}