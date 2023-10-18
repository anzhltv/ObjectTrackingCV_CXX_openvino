#ifndef TRALG_H_ 
#define TRALG_H_
#include "CallHistogram.h"
#include "EuclideanDistTracker.h"
#include "NetOpenVino.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>


class TrackingAlgorithm 
{
public:
    TrackingAlgorithm() : arrayID(100), idSave(-1), report(true), countFrameCam(0) {};
    int idSave; // предыдущий id объекта
    bool report; // Наличие объекта в камере
    int countFrameCam; //Количество кадров проверки объекта
    
    /*
    метод для обновления трекера, получения векторов, поиск сравнения, выполнение алгоритма
    Input:
    detections - координаты бокса , numCam - номер камеры, frame -  кадр, countSame - количество одинаковых объектов, vectorHist - сохраненные гистограммы , tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
    Output:
    бокс с верно определнным айди на кадре
    */
    void updateCameraTracking(std::vector<cv::Rect> &detections, int numCam, cv::Mat frame, int& countSame, std::vector<std::vector<float>>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg, NeuralNetworkDetector& detectOnnx);
    
    std::vector<int> arrayID; // Массив для накопления совпадений с конкретным объектом
private:
    int idCorrect; //финально определенный id для вывода на бокс

    /*
    метод для определения нового объекта на полученном кадре
    Input:
    idsBoxes - координаты бокса и id нового объекта, numCam - номер камеры, frame - сам кадр, countSame - количество одинаковых объектов,
    vectorHist - сохраненные гистограммы, tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
    Output:
    верно определенный id объекта и бокс на кадре
    */
    void CameraTracking(const std::vector<std::vector<int>>& idsBoxes, int numCam, const cv::Mat& frame, int& countSame,
        std::vector<std::vector<float>>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg, NeuralNetworkDetector& detectorOnnx);

    /*
    метод для определения корректного айди для объекта
    если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
    то отдаем объекту найденный айди,
    иначе айди по порядку
    Input:
    idGlobal - айди текущего объекта
    Output:
    корректный айди объекта
    */
    auto FindMaxSameId(int idGlobal);

    /*
    метод на случай, если найден тот же объект
    если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
    то очищаем текущий элемент с собранным вектором для этого объекта, так как он уже есть
    + увеличиваем количество совпадающих объектов
    Input:
    arrayHist - массив с накопленными гистограммами, countSame - подсчет одинаковых элементов, idSave - айди предыдущего объекта
    Output:
    количество одинаковых элементов
    */
    int SameObjectCore(std::vector<std::vector<float>>& arrayHist, int countSame, int idSave, std::vector<int>& arrID);
    int SameObject(std::vector<std::vector<float>>& arrayHist, int countSame, int idSave);

    /*
    перегрузка для метода на случай, если найден тот же объект, для второй камеры, в этом случае берем массив айди из другого объекта класса
    если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
    то очищаем текущий элемент с собранным вектором для этого объекта, так как он уже есть
    + увеличиваем количество совпадающих объектов
    Input:
    arrayHist - массив с накопленными гистограммами, countSame - подсчет одинаковых элементов, idSave - айди предыдущего объекта, arrID - массив содержащий совпадения с существующими объектами для второй камеры 
    Output:
    количество одинаковых элементов
    */
    int SameObject(std::vector<std::vector<float>>& arrayHist, int countSame, int idSave, std::vector<int>& arrID);

    /*
    метод для увеличения countSame - если появился новый объект, а старый был определен к уже существующим
    Input:
    id текущего объекта по порядку, numCam, numCam2 - номер текущей и другой камеры, countSame - количество одинаковых объектов, arrayHist - сохраненные гистограммы, trackAlg - объект класса алгоритма с другой камеры
    Output:
    заполненный массив arrayID
    */
    void NewObject(int id, int numCam, int numCam2, int& countSame, std::vector<std::vector<float>>& arrayHist, TrackingAlgorithm& trackAlg);
    
    //время с начала последнего найденного объекта
    std::chrono::steady_clock::time_point timeStart;
    //время с начала нового найденного объекта
    std::chrono::steady_clock::time_point timeEnd;
    //сохранение центра последнего найденного объекта
    cv::Point2i centerPoint;
};

#endif //TRALG