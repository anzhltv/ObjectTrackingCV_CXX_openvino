#ifndef UTILS_H_ 
#define UTILS_H_

#include <iostream>
#include <opencv2/opencv.hpp>

constexpr auto BORDER_Y = 50;
constexpr auto Y_PLUS = 30;
constexpr auto Y_MINUS = 15;

// Метод для вычисления площади прямоугольника
// Вход:
// x1, y1, x2, y2 - координаты прямоугольника
// Выход:
// Площадь прямоугольника
int CalculateArea(int x1, int y1, int x2, int y2) 
{
    return (x2 - x1) * (y2 - y1);
}

// Метод для коррекции отображения ID
// Если прямоугольник касается верхней границы, текст отображается под прямоугольником, иначе над ним
// Вход:
// y, h - координата левого верхнего угла и высота прямоугольника
// Выход:
// Скорректированная координата y
int IfBorder(int y, int h) 
{
    return (y < BORDER_Y) ? (y + h + Y_PLUS) : (y - Y_MINUS);
}


// Метод для нахождения центральной точки
// Вход:
// Координаты прямоугольника
// Выход:
// Координаты центра
cv::Point2i CenterPointSave(int x1, int x2, int y1, int y2)
{       int cx = (x1 + x1 + x2) / 2;
        int cy = (y1 + y1 + y2) / 2;
    return cv::Point2i(cx, cy);
}

#endif //UTILS_H_