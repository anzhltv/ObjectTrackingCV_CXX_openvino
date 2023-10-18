#ifndef ECL_H_ 
#define ECL_H_

#include <vector>
#include <map>
#include <opencv2/core.hpp>

class EuclideanDistTracker 
{
public:
    int idCount; //подсчет объектов в камере
    int saveId;
    EuclideanDistTracker();
    /*
    метод для трекинга на одном кадре
    Input
    objectsRect - x,y,w,h объекта
    Output
    координаты объекта и верно определенный id объекта
    */
    std::vector<std::vector<int>> update(const std::vector<cv::Rect>& objectsRect, cv::Point2i cxy, long long deltaTime); //обновление трекера 
private:
    std::map<int, cv::Point2i> centerPoints;
    
};
#endif