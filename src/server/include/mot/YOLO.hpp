/* @Author: YueLin */

/* Include standrad libraries */
#include <cmath>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>

/* Include third-party libraries */
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "cuda_runtime_api.h"

/* Include custom libraries */
#include "logger"
#include "mot/object.hpp"

namespace det
{
    /* TensorRT names */
    char *INPUT = (char*)"images", *OUTPUT = (char*)"outputs";

    /* YOLO algorithm */
    class YOLO
    {
        private:
            void* buffers[2];
            int input, output;
            float *image, *detection;
            nvinfer1::IExecutionContext* context;
        
        private:
            int nc, dim, SIZE;
            double nms_threshold, score_threshold;
        
        public:
            ~YOLO();
            YOLO(std::string, double*, int*);
        
        public:
            Objects detect(cv::Mat&, double);
            Objects detect(cv::Mat& img) {return detect(img, score_threshold);}
        
        private:
            void inference();
            Objects proposals(double);
            std::vector<int> nms(Objects&);
            void preprocess(cv::Mat&, float*);
            Objects post_process(float*, const int, const int);
    };

    /* Free memory */
    YOLO::~YOLO()
    {
        delete context;
    }

    /* Initialize YOLO model */
    YOLO::YOLO(std::string path, double* thresholds, int* info)
    {
        /* Initialize */
        size_t size{0};
        cudaSetDevice(0);
        char *stream{nullptr};
        static Logger _logger;

        /* Thresholds */
        nms_threshold = thresholds[0];
        score_threshold = thresholds[1];

        /* About model */
        SIZE = info[0]; nc = info[1]; dim = info[2];

        /* Apply for memory from CPU */
        image = new float[3 * SIZE * SIZE];
        detection = new float[dim * (nc + 5)];

        /* Load tensorrt file */
        std::ifstream file(path, std::ios::binary);
        file.seekg(0, file.end); size = file.tellg();
        file.seekg(0, file.beg); stream = new char[size];
        file.read(stream, size); file.close();

        /* The output of YOLOv5 is 'output0' */
        if(path.find("v5") != std::string::npos) OUTPUT = (char*)"output0";

        /* Make tensorrt context and engine */
        context = nvinfer1::createInferRuntime(_logger)
               -> deserializeCudaEngine(stream, size)
               -> createExecutionContext();
        const nvinfer1::ICudaEngine& engine = context->getEngine();

        /* Set bindding indexes */
        const int tensors = engine.getNbIOTensors();
        for(int n = 0; n < tensors; n++)
        {
            const char* name = engine.getIOTensorName(n);
            if(!std::strcmp(name, INPUT)) input = n;
            else if(!std::strcmp(name, OUTPUT)) output = n;
            else throw "Unexpected tensor name!";
        }
    }

    /* Perform objects detection once */
    Objects YOLO::detect(cv::Mat& frame, double threshold)
    {
        float padding[2];
        double st = score_threshold;
        score_threshold = threshold;
        const int w = frame.cols, h = frame.rows;
        preprocess(frame, padding); inference();
        Objects objects = post_process(padding, w, h);
        score_threshold = st; return objects;
    }

    /* Do inference */
    void YOLO::inference()
    {
        /* Apply for memory from cuda */
        cudaMalloc(&buffers[input], 3 * SIZE * SIZE * sizeof(float));
        cudaMalloc(&buffers[output], dim * (nc + 5) * sizeof(float));
        
        /* Create cuda stream */
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        /* Copy data to cuda */
        cudaMemcpyAsync(
            buffers[input], image,
            3 * SIZE * SIZE * sizeof(float),
            cudaMemcpyHostToDevice, stream
        );

        /* Start inference */
        context->setTensorAddress(OUTPUT, buffers[output]);
        context->setTensorAddress(INPUT, buffers[input]);
        context->enqueueV3(stream);

        /* Copy results to cpu */
        cudaMemcpyAsync(
            detection, buffers[output],
            dim * (nc + 5) * sizeof(float),
            cudaMemcpyDeviceToHost, stream
        );

        /* Wait for inference to complete */
        cudaStreamSynchronize(stream);

        /* Release memory and stream */
        cudaFree(buffers[input]);
        cudaFree(buffers[output]);
        cudaStreamDestroy(stream);
    }

    /* Resize */
    void YOLO::preprocess(cv::Mat& frame, float* padding)
    {
        /* Convert from BGR mode into RGB mode */
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        /* Resize image */
        int width, height;
        const int w = frame.cols, h = frame.rows;
        if(std::max(w, h) < SIZE)
        {
            width = (SIZE - w) >> 1;
            height = (SIZE - h) >> 1;
        }
        else
        {
            cv::Size size;
            if(h > w)
            {
                size = cv::Size(std::round(SIZE * w / h), SIZE);
                width = (SIZE - size.width) >> 1; height = 0;
            }
            else
            {
                size = cv::Size(SIZE, std::round(SIZE * h / w));
                height = (SIZE - size.height) >> 1; width = 0;
            }
            cv::resize(frame, frame, size);
        }

        /* Padding */
        if(height || width)
            cv::copyMakeBorder(
                frame, frame, height, height, width, width, 
                cv::BORDER_CONSTANT, cv::Scalar::all(0x72)
            );
        if(frame.cols != SIZE || frame.rows != SIZE)
            cv::resize(frame, frame, cv::Size(SIZE, SIZE));
        padding[0] = height; padding[1] = width;

        /* Normalization */
        for(int c = 0; c < 3; c++)
            for(int y = 0; y < SIZE; y++)
                for(int x = 0; x < SIZE; x++)
                    image[c * SIZE * SIZE + y * SIZE + x] 
                    = frame.at<cv::Vec3b>(y, x)[c] / 255.f; 
    }

    /* Generate proposals */
    Objects YOLO::proposals(double threshold)
    {
        Objects proposals;
        for(int b = 0; b < dim; b++)
        {
            /* Get confidence */
            int p = b * (nc + 5);
            double object = detection[p + 4];

            /* Kepp target with confidence > threshold */
            if(object > threshold)
            {
                /* Get box */ float
                x = detection[p + 0], y = detection[p + 1],
                w = detection[p + 2], h = detection[p + 3];

                /* Center coordinate to upper left corner coordinate */
                x -= w / 2; y -= h / 2;

                /* Get category */
                int cls = 0;
                float score = 0;
                for(int c = 0; c < nc; c++)
                    if(detection[p + c + 5] > score)
                    {
                        cls = c;
                        score = detection[p + c + 5];
                    }
                score *= object;

                /* Create box */
                if(score > threshold)
                {
                    Object obj;
                    obj.rect.x = x;
                    obj.rect.y = y;
                    obj.label = cls;
                    obj.prob = score;
                    obj.rect.width = w;
                    obj.rect.height = h;
                    proposals.push_back(obj);
                }
            }
        }
        return proposals;
    }

    /* non-maximum supprtssion */
    std::vector<int> YOLO::nms(Objects& proposals)
    {
        std::vector<int> result;
        const int num = proposals.size();
        
        /* Sort boxes according to confidence */
        if(!proposals.empty())
            sorted(proposals, 0, num - 1);
        
        /* Calculate the area of boxes */
        float* s = new float[num];
        for(int b = 0; b < num; b++)
            s[b] = proposals[b].rect.area();
        
        /* Filter */
        result.clear();
        for(int i = 0; i < num; i++)
        {
            bool keep = true;
            int n = result.size();
            Object& obj1 = proposals[i];
            for(int j = 0; j < n; j++)
            {
                Object& obj2 = proposals[result[j]];
                float s0 = (obj1.rect & obj2.rect).area();
                if(s0 > nms_threshold * (s[i] + s[result[j]] - s0))
                {
                    keep = false; break;
                }
            }
            if(keep) result.push_back(i);
        }
        return result;
    }

    /* Decode outputs and non-maximum supprtssion */
    Objects YOLO::post_process(float* padding, const int w0, const int h0)
    {
        /* Generate proposals */
        Objects proposal = proposals(score_threshold);

        /* non-maximum supprtssion */
        std::vector<int> keep = nms(proposal);
        const int num = keep.size();

        /* Generate boxes */
        Objects objects(num); float
        w = w0 / (SIZE - padding[1] * 2),
        h = h0 / (SIZE - padding[0] * 2);
        for(int b = 0; b < num; b++)
        {
            /* Remap rect */
            Object obj = proposal[keep[b]];
            obj.rect.x -= padding[1];
            obj.rect.y -= padding[0];
            obj.rect.height *= h;
            obj.rect.width *= w;
            obj.rect.x *= w;
            obj.rect.y *= h;
            objects[b] = obj;
        }
        return objects;
    }
}
