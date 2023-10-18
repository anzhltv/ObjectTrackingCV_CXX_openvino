#ifndef OPRT_H_ 
#define OPRT_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>


/* 
Перегрузка оператора +, для сложения двух векторов
Input
vec1, vec2 - два входных вектора 
Output
результат сложения векторов
*/
template <typename T>
std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2)
{
    std::vector<T> result;
    if (vec1.size() != vec2.size())
        return result;
    result.resize(vec1.size());
    auto vec1_it = vec1.cbegin(),
        vec2_it = vec2.cbegin();
    std::for_each(result.begin(), result.end(), [&vec1_it, &vec2_it](T& t)->void {
        t = (*(vec1_it++)) + (*(vec2_it++));
        });
    return result;
}


/*
Перегрузка оператора *, для умножения вектора на число
Input
vec - входной вектор, a - число на которое нужно умножить вектор 
Output
результат умножения вектора на число 
*/
std::vector<float> operator *(const std::vector<float>& vec, const float a)
{
    std::vector<float> result;
    result.resize(vec.size());
    std::copy(vec.cbegin(), vec.cend(), result.begin());
    for (auto& t : result)
        t *= a;
    return result;
}

#endif