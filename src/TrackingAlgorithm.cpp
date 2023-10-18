#include "TrackingAlgorithm.h"
#include "SmallUtils.h"
#include "VectorOnnx.h"


constexpr auto COUNT_FRAME = 40; // Количество кадров для идентификации человека
const std::vector<float> OPT_PARAM = { 0.5f, 0.21f }; // Границы сравнения векторов для каждого кадра
constexpr auto PART_FRAME = 6; // Часть от общего числа кадров проверки
const std::vector<float> PARAM = { 0.65f, 0.48f }; // Границы сравнения векторов для каждого кадра


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
auto TrackingAlgorithm::FindMaxSameId(int idGlobal) 
{
    std::vector<int> arrayID_list(arrayID.begin(), arrayID.end());
    // Если максимальное совпадение больше чем 1/6 от общего числа кадров проверки
    if (*std::max_element(arrayID_list.begin(), arrayID_list.end()) >= COUNT_FRAME / PART_FRAME)
    {
        // То берем найденный id
        idGlobal = static_cast<int>(std::distance(arrayID_list.begin(), std::max_element(arrayID_list.begin(), arrayID_list.end())));
        
    }
    return idGlobal;
}


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
int TrackingAlgorithm::SameObjectCore(std::vector<std::vector<float>>& arrayHist, int countSame, int idSave, std::vector<int>& arrID)
{
    //если был найден существующий объект, то очистка собранного вектора и массива id + увеличение числа совпадающих объектов
    std::vector<int>::iterator maxEl = std::max_element(arrID.begin(), arrID.end());
    if (*maxEl >= COUNT_FRAME / PART_FRAME)
    {
        arrayHist[idSave - countSame] = cv::Mat();
        ++countSame;
    }
    std::for_each(arrID.begin(), arrID.end(), [](int& value) 
        {
        value = 0;
        }
    );
    return countSame;
}


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
int TrackingAlgorithm::SameObject(std::vector<std::vector<float>>& arrayHist, int countSame, int idSave)
{
    return SameObjectCore(arrayHist, countSame, idSave, arrayID);
}


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
int TrackingAlgorithm::SameObject(std::vector<std::vector<float>>& arrayHist, int countSame, int idSave, std::vector<int>& arrID)
{
    return SameObjectCore(arrayHist, countSame, idSave, arrID);
}


/*
метод для увеличения countSame - если появился новый объект, а старый был определен к уже существующим
Input:
id текущего объекта по порядку, numCam, numCam2 - номер текущей и другой камеры, countSame - количество одинаковых объектов, arrayHist - сохраненные гистограммы, trackAlg - объект класса алгоритма с другой камеры
Output:
заполненный массив arrayID
*/
void TrackingAlgorithm::NewObject(int id, int numCam, int numCam2, int& countSame, std::vector<std::vector<float>>& arrayHist, TrackingAlgorithm& trackAlg)
{
    // Если новый объект
    if (id != idSave) 
    {
        // Проверяем содержимое массива ID для первой камеры и увеличиваем количество одинаковых объектов
        countSame = SameObject(arrayHist, countSame, idSave);

        if (trackAlg.report) 
        {
            // Проверяем содержимое массива ID для второй камеры и увеличиваем количество одинаковых объектов
            countSame = SameObject(arrayHist, countSame, trackAlg.idSave, trackAlg.arrayID);
        }
        // Обновляем счетчик кадров для гистограмм и для детектора
        countFrameCam = COUNT_FRAME;
    }
}


/*
метод для определения айди нового объекта на полученном кадре
Input:
idsBoxes - координаты бокса и id нового объекта, numCam - номер камеры, frame - сам кадр, countSame - количество одинаковых объектов, 
vectorHist - сохраненные гистограммы, tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
Output:
верно определенный id объекта и бокс на кадре
*/
void TrackingAlgorithm::CameraTracking(const std::vector<std::vector<int>> &idsBoxes, int numCam, const cv::Mat &frame, int& countSame,
    std::vector<std::vector<float>>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg, NeuralNetworkDetector& detectorOnnx)
{
    auto numCam2 = (numCam + 1) % 2;

    for (const auto & box_id : idsBoxes) 
    {
        auto x = box_id[0];
        auto y = box_id[1];
        auto w = box_id[2];
        auto h = box_id[3];
        auto id = box_id[4];

        auto y1 = IfBorder(y, h);
        report = false;
        cv::Mat framePlt;
            auto X = w-x;
            if (x + w > frame.cols) 
                X = frame.cols-x;
            auto Y = h-y;
            if (h + y > frame.rows)
                Y = frame.rows-y;
            if (y < 0)
                y = 0;

            framePlt = frame(cv::Rect(x, y, X, Y));
            NewObject(id, numCam, numCam2, countSame, vectorHist, trackAlg);
            auto idGlobal = id - countSame;
            idSave = id;
            if (countFrameCam > 0)
            {
                //CallHistogram(framePlt, vectorHist, idGlobal, OPT_PARAM[numCam], arrayID);
                CountCompare(detectorOnnx, framePlt, vectorHist, idGlobal, PARAM[numCam], arrayID);
                countFrameCam -= 1;
                idCorrect = idGlobal;
            }
            else
            {
                idCorrect = FindMaxSameId(idGlobal);
                id = idCorrect;
                putText(frame, "Object " + std::to_string(id), cv::Point(x, y1), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
            }
            cv::rectangle(frame, cv::Point(x, y), cv::Point(w, h), cv::Scalar(255, 0, 0), 3);
            tracker[numCam2].idCount = tracker[numCam].idCount;
            tracker[numCam2].saveId = tracker[numCam].saveId;
            timeStart = std::chrono::steady_clock::now();
            centerPoint = CenterPointSave(x, w, y, h);
    }
}


/*
метод для обновления трекера, получения векторов, поиск сравнения, выполнение алгоритма
Input:
detections - координаты бокса , numCam - номер камеры, frame -  кадр, countSame - количество одинаковых объектов, vectorHist - сохраненные гистограммы , tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры, param - параметр сранвения векторов,  detectOnnx - объект класса NeuralNetworkDetector для поиска вектора бокса 
Output:
бокс с верно определенным айди на кадре
*/
void TrackingAlgorithm::updateCameraTracking(std::vector<cv::Rect> &detections, int numCam, cv::Mat frame, int& countSame, std::vector<std::vector<float>>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg, NeuralNetworkDetector& detectOnnx)
{   
    timeEnd = std::chrono::steady_clock::now();
    auto deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
    auto deltaTimeCount = deltaTime.count();
    report = true;
    // Обновление трекера
    auto idsBoxes1 = tracker[numCam].update(detections, centerPoint, deltaTimeCount);
    // Трекинг, назначение ID, отрисовка боксов
    CameraTracking(idsBoxes1, numCam, frame, countSame, vectorHist,  tracker, trackAlg, detectOnnx);
}