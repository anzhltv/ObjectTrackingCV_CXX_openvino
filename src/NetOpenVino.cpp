#include "NetOpenVino.h"

/*
метод для загрузки сети и получения информации о входе и выходе сети
Input
modelFormat - формат модели,  modelPath - путь до модели сети, device - устройство CPU или GPU,  size - размер входа сети
Output
модель сети
*/
void NeuralNetworkDetector::ReadNet(const std::string& modelFormat, const std::string& modelPath, const std::string& device, const cv::Size& size)
{
    ie = InferenceEngine::Core();
    if (modelFormat == "openvino") {
        network = ie.ReadNetwork(modelPath + ".xml");
    }
    else if (modelFormat == "onnx") {
        try {
            network = ie.ReadNetwork(modelPath + ".onnx");
        }
        catch (const InferenceEngine::Exception& ex) {
            std::cerr << "error:" << ex.what();
        }
    }
    std::cout << "Loading model to the device" << std::endl;
    sizeNet = size;
    outputs_info = network.getOutputsInfo();
    inputs_info = network.getInputsInfo();
    batch_size = network.getBatchSize();
    deviceNet = device;
}


/*
метод для подготовки входных и выходных блобов
Input
modelFormat - формат модели,  modelPath - путь до модели сети, device - устройство CPU или GPU,  size - размер входа сети
Output
модель сети
*/
void NeuralNetworkDetector::getInputOutputInfo()
{   //подготовка входного блоба
    std::tie(input_name, input_info) = *inputs_info.begin();
    // Set input layout and precision
    input_info->setLayout(InferenceEngine::Layout::NCHW);
    input_info->setPrecision(InferenceEngine::Precision::U8);
    //подготовка выходного блоба
    std::tie(output_name, output_info) = *outputs_info.begin();

    const InferenceEngine::SizeVector output_shape = output_info->getTensorDesc().getDims();
    max_proposal_count = output_shape[2];
    object_size = output_shape[3];
    output_info->setPrecision(InferenceEngine::Precision::FP32);

    //загрузка модели
    executableNetwork = ie.LoadNetwork(network, deviceNet);
    //создание запроса 
    inferRequest = executableNetwork.CreateInferRequest();
}


/*
метод для получения выходных данных сети
Input
Frame - входное изображение
Output
detection - выходные данные сети
*/
const float* NeuralNetworkDetector::Forward(const cv::Mat& Frame)
{
    cv::Mat image = Frame;
    InferenceEngine::Blob::Ptr input = inferRequest.GetBlob(input_name);
    //подготовка входных данных
    cv::Mat orig_image = image.clone();
    for (size_t b = 0; b < batch_size; b++) {
        InferenceEngine::SizeVector blobSize = input->getTensorDesc().getDims();
        const size_t width = blobSize[3];
        const size_t height = blobSize[2];
        const size_t channels = blobSize[1];
        InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
        auto mblobHolder = mblob->wmap();
        uint8_t* blob_data = static_cast<uint8_t*>(mblobHolder);
        cv::Mat resized_image(orig_image);
        cv::resize(orig_image, resized_image, sizeNet);

        int batchOffset = b * width * height * channels;

        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    blob_data[batchOffset + c * width * height + h * width + w] = resized_image.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    }
    //выполнение инференса
    inferRequest.Infer();
    //получение выходных данных
    InferenceEngine::Blob::Ptr output = inferRequest.GetBlob(output_name);
    InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
    //преобразование в const float*
    auto moutputHolder = moutput->rmap();
    const float* detection = moutputHolder.as<const float*>();
    return detection;
}



/*
метод для преобразования выходных данных сети openvino
Input
detection выходные данные сети,  detections - для записи результата,  image - изображение для получения размера, size - минимальный размер бокса, sizeA - максимальный размер бокса
Output
detections - выходные данные сети в формате вектора 4 точек
*/
void NeuralNetworkDetector::outputOpenVino(const float* detection, std::vector<cv::Rect>& detections, const cv::Mat& image, int size, int sizeA)
{
    for (size_t cur_proposal = 0; cur_proposal < max_proposal_count; cur_proposal++) {
        float image_id = detection[cur_proposal * object_size + 0];
        float label = detection[cur_proposal * object_size + 1];
        float confidence = detection[cur_proposal * object_size + 2];

        if (image_id < 0 || confidence == 0.0f) {
            continue;
        }
        if (confidence > 0.3f) {
            float xmin = detection[cur_proposal * object_size + 3] * image.cols;
            float ymin = detection[cur_proposal * object_size + 4] * image.rows;
            float xmax = detection[cur_proposal * object_size + 5] * image.cols;
            float ymax = detection[cur_proposal * object_size + 6] * image.rows;
            auto area = (xmax - xmin) * (ymax - ymin);
            if (area > size && area < sizeA) {
                cv::Rect rect(xmin, ymin, xmax, ymax);
                detections.push_back(rect);
            }
        }
    }
}