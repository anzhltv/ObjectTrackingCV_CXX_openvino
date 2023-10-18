#include "FindCont.h"
#include "TrackingAlgorithm.h"
#include "NetOpenVino.h"
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <chrono>
#include <vector>
#include <iostream>

// параметры минимальных размеров боксов
constexpr auto PERSENT_SIZE_BOX_L = 0.045;
constexpr auto PERSENT_SIZE_BOX_R = 0.0076;
// параметры для обрезки кадра со второй камеры
constexpr auto PARAM_ROI_B = 4.2;
constexpr auto PARAM_ROI_E = 2.5;
// номера камер
constexpr int NUM_CAM[] = { 0, 1 };
// параметры для отображения видео
constexpr auto PARAM_ROI_W = 1.3;
constexpr auto PARAM_ROI_H = 0.7;
//размеры изображений для сетей
const cv::Size inputSizeOpenvino(300, 300); // Размер входного изображения
const cv::Size inputSizeONNX(128, 256); // Размер входного изображения

constexpr auto DEVICE = "CPU"; // Устройство для выполнения инференса (GPU или CPU)


int main()
{
    constexpr auto modelFormatOpenvino = "openvino"; // Формат модели (openvino или onnx)
    constexpr auto modelFormatONNX = "onnx"; // Формат модели (openvino или onnx)

    constexpr auto modelPathOpenvino = "C:/net/GeneralNMHuman_v1.0_IR10_FP16"; // Путь к файлам модели openvino (без расширений)
    constexpr auto modelPathONNX = "C:/net/original_reid"; // Путь к файлам модели onnx (без расширений)

    std::vector<EuclideanDistTracker> tracker(2); // Создание двух объектов класса EuclideanDistTracker для трекинга
    std::vector<TrackingAlgorithm> trackAlg(2); // Создание двух объектов класса TrackingAlgorithm для выполнения алгоритма
   
    constexpr auto path1 = "C:/video/Camera3.avi";
    constexpr auto path2 = "C:/video/Camera4.avi";

    //создания объекта для работы с моделью openvino
    NeuralNetworkDetector detectorOpenvino;
    //чтение модели openvino
    detectorOpenvino.ReadNet(modelFormatOpenvino, modelPathOpenvino, DEVICE, inputSizeOpenvino);
    //создания объекта для работы с моделью onnx
    NeuralNetworkDetector detectorOnnx;
    //чтение модели onnx
    detectorOnnx.ReadNet(modelFormatONNX, modelPathONNX, DEVICE, inputSizeONNX);

    //получение входной и выходной информации
    detectorOpenvino.getInputOutputInfo();
    detectorOnnx.getInputOutputInfo();
    std::vector<float> vectorOnnx;


    cv::Mat input_frame;

    cv::VideoCapture cap1(path1);
    cv::VideoCapture cap2(path2);
    cv::Mat frame1_1, frame2_2; // Кадры для первой и второй камеры

    cap1.read(frame1_1);
    cap2.read(frame2_2);
    
    const int width = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_HEIGHT));

    frame2_2 = frame2_2(cv::Rect(int(width / PARAM_ROI_B), 0, int(width / PARAM_ROI_E), height)); // Обрезка кадра

    const auto size1 = static_cast<int>(PERSENT_SIZE_BOX_L * width * height); // Минимальные размеры боксов
    const auto size = static_cast<int>(width * height / 3);
    const auto size2 = static_cast<int>(PERSENT_SIZE_BOX_R * width * height);

    std::vector<cv::Mat> vector_hist(100);
    std::vector<std::vector<float>> vectorAll(100);
    auto count_same = 0; // Переменная для подсчета одинаковых объектов

    if (!cap1.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
    }
    int countFrame = 0, div = 5;
    while (cap1.isOpened())
    {
        countFrame += 1;
        setlocale(LC_ALL, "Russian");
        cv::Mat frame1, mask1, mask2;
        cv::Mat frame2, roi2;
        bool isSuccess1 = cap1.read(frame1);
        bool isSuccess2 = cap2.read(frame2);

        roi2 = frame2(cv::Rect(int(width / PARAM_ROI_B), 0, int(width / PARAM_ROI_E), height));
        // 
        //if (countFrame == 30)
        //    countFrame = 0;
        //if (countFrame % div != 0)
        //    continue;


        /*GettingCoordinates
        метод для для объединения метода Substractor и DetectContour
        Input:
        frame1 - кадр с которым производится вычитание, frame2 - новый кадр , kern - размер ядра, iter - количество итераций, size - минимальный размер бокса
        Output:
        заполненный вектор detections
        */
        std::vector<cv::Rect> detections1, detections2;
        const float* detectionFloat1 = detectorOpenvino.Forward(frame1);
        detectorOpenvino.outputOpenVino(detectionFloat1, detections1, frame1, size1, size);

        const float* detectionFloat2 = detectorOpenvino.Forward(roi2);
        detectorOpenvino.outputOpenVino(detectionFloat2, detections2, roi2, size2, size);

        /*
        метод для обновления трекера, получения векторов, поиск сравнения, выполнение алгоритма
        Input:
        detections - координаты бокса , numCam - номер камеры, frame -  кадр, countSame - количество одинаковых объектов, vectorHist - сохраненные гистограммы , tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
        Output:
        бокс с верно определнным айди на кадре
        */
        trackAlg[0].updateCameraTracking(detections1, NUM_CAM[0], frame1, count_same, vectorAll, tracker, trackAlg[1], detectorOnnx);
        trackAlg[1].updateCameraTracking(detections2, NUM_CAM[1], roi2, count_same, vectorAll, tracker, trackAlg[0], detectorOnnx);


        if (isSuccess1 && isSuccess2)
        {
            cv::Mat Combi;
            cv::hconcat(frame1, frame2, Combi);
            cv::resize(Combi, Combi, cv::Size(width * PARAM_ROI_W, height * PARAM_ROI_H));
            cv::imshow("Combined", Combi);
        }

        if (!isSuccess1 && !isSuccess2)
        {
            std::cout << "End of video" << std::endl;
            break;
        }

        int key = cv::waitKey(1);
        if (key == 'q')
        {
            std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
            break;
        }
    }

    return 0;
}