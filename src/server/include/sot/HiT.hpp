/* @Author: YueLin */

/* Include standrad libraries */
#include <cmath>
#include <chrono>
#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>

/* Include third-party libraries */
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "cuda_runtime_api.h"

/* Include custom library */
#include "logger"

/* Get the current time (us) */
#define now std::chrono::duration_cast<std::chrono::microseconds>(\
    std::chrono::system_clock::now().time_since_epoch()\
).count

/* Macro functions */
#define CLIP do{\
    using namespace std;\
    boxes[n].x = min(max(0.0, x1), w - margin);\
    boxes[n].y = min(max(0.0, y1), h - margin);\
    boxes[n].width = max(min(max(margin, x2), w) - boxes[n].x, margin);\
    boxes[n].height = max(min(max(margin, y2), h) - boxes[n].y, margin);\
}while(false)
#define MALLOC(idx, sz, c) cudaMalloc(\
    &buffers[indexes[idx]],\
    c * sz * sz * sizeof(float)\
)
#define CPU2CUDA(idx, arr, sz, c) cudaMemcpyAsync(\
    buffers[indexes[idx]], arr,\
    c * sz * sz * sizeof(float),\
    cudaMemcpyHostToDevice, stream\
)
#define CUDA2CPU(idx, arr, sz, c) cudaMemcpyAsync(\
    arr, buffers[indexes[idx]],\
    c * sz * sz * sizeof(float),\
    cudaMemcpyDeviceToHost, stream\
)

namespace sot
{
    typedef cv::Rect2f Box;

    /* Constants */
    const int N = 3;
    const int SIZE1 = 0x80, SIZE2 = 0x100;

    const int SZ = SIZE2 >> 4;
    const int SZ2 = SZ * SZ;
    const float PI = std::acos(-1);
    const float STD[3] = {0.229, 0.224, 0.225};
    const float MEAN[3] = {0.485, 0.456, 0.406};
    const char* NAME[6] = {
        "search", "template",
        "score_map", "size_map", "offset_map", "mlp_score"
    };
    
    enum Index{SEARCH = 0, TEMPLATE, SCORE, SIZE, OFFSET, MLP, M_IDX};

    /* List for cv::Rect2f */
    class BoxArrays
    {
        public:
            int len[N];                  // List length

        private:
            int length;
            int pointers[N];             // Current position
            std::vector<Box> arrays[N];  // Array of cv::Rect2f

        /* Public methods */
        public:
            BoxArrays(){}
            void resize(int);
            Box begin(int), end(int);
            void append(int, cv::Rect), clear(int);
    };

    /* HiT algorithm */
    class HiT
    {
        public:
            int targets;     // Number of target
            Box boxes[N];    // Target's position
            bool lost[N];    // Whether the target is lost
            double last[N];  // Time of last appearance of targets
            double retrieval_threshold, wait;  // For retrieval

        private:
            /* For motion tracking */
            int momentum;        // Maximum number of motion predictions
            bool motion[N];      // Whether to enable motion tracking
            bool retrieval[N];   // Whether to enable global retrieval
            BoxArrays history;   // History box lists
            int s0[N], wid0[N];  // The area of initial boxes

            /* For HiT inference */
            int indexes[M_IDX];
            void* buffers[M_IDX];
            cv::Mat templates[N];
            float temp[N][3 * SIZE1 * SIZE1];
            float search[N][3 * SIZE2 * SIZE2];
            float weights[SZ2], predicts[N][4];
            nvinfer1::IExecutionContext* context;
            float mlp[N][SZ2], sizes[N][SZ2 << 1];
            float scores[N][SZ2], offsets[N][SZ2 << 1];
        
            /* Hyperparameters */
            double area, alpha, beta, margin;
            double mlp_threshold, score_threshold;

        /* Constructor and destructor */
        public:
            ~HiT();
            HiT(std::string, double*);
        
        /* Public methods */
        public:
            void track(cv::Mat&);
            void init(cv::Mat&, cv::Rect*, int);
            void match(cv::Mat&, std::vector<Box>&, bool*);
        
        /* Private methods */
        private:
            bool decode(const int);
            void sample(cv::Mat&, cv::Mat*, int, float*);
            void inference(float*, const int, const int);
            void preprocess(cv::Mat&, bool, float**, float*);
            void post_process(float*, const int, const int, const int);
    };

    /* Free memory */
    HiT::~HiT()
    {
        delete context;
    }

    /* Initialize HiT model */
    HiT::HiT(std::string path, double* params)
    {
        /* Initialize */
        size_t size{0};
        cudaSetDevice(0);
        float weight[SZ];
        char *stream{nullptr};
        static Logger _logger;

        /* Hyperparameters */
        wait = params[8];
        mlp_threshold = params[4];
        score_threshold = params[6];
        retrieval_threshold = 1 - params[7];
        beta = params[0]; alpha = params[1];
        area = params[5]; margin = params[2];
        history.resize(momentum = params[3]);

        /* Load tensorrt file */
        std::ifstream file(path, std::ios::binary);
        file.seekg(0, file.end); size = file.tellg();
        file.seekg(0, file.beg); stream = new char[size];
        file.read(stream, size); file.close();

        /* Make tensorrt context and engine */
        context = nvinfer1::createInferRuntime(_logger)
               -> deserializeCudaEngine(stream, size)
               -> createExecutionContext();
        const nvinfer1::ICudaEngine& engine = context->getEngine();

        /* Set bindding indexes */
        assert(engine.getNbIOTensors() == M_IDX);
        for(int i = 0; i < M_IDX; i++)
        {
            const char* name = engine.getIOTensorName(i);
            for(int n = 0; n <= M_IDX; n++)
            {
                if(n == M_IDX) throw "Unexpected tensor name!";
                if(!std::strcmp(name, NAME[n]))
                {
                    indexes[n] = i; break;
                }
            }
        }

        /* Initialize output weights */
        for(int w = 0; w < SZ; w++)
            weight[w] = 0.5 * (1 - std::cos(2 * PI * (w + 1) / (SZ + 1)));
        for(int v = 0; v < SZ; v++)
            for(int w = 0; w < SZ; w++)
                weights[SZ * v + w] = weight[v] * weight[w];
    }

    /* Perform objects tracking once */
    void HiT::track(cv::Mat& frame)
    {
        float resize[N];
        int h0 = frame.rows, w0 = frame.cols;
        preprocess(frame, true, (float**)search, resize);
        inference(resize, h0, w0); post_process(resize, h0, w0, targets - 1);
    }

    /* Initialize templates and motion tracking */
    void HiT::init(cv::Mat& image, cv::Rect* rects, int num)
    {
        /* Initialize tracking template */
        for(targets = 0; targets < num; targets++)
        {
            boxes[targets] = rects[targets];
            templates[targets] = image(rects[targets]).clone();
        }
        preprocess(image, false, (float**)temp, nullptr);

        /* For the motion track */
        for(int t = 0; t < num; t++)
        {
            history.clear(t);
            motion[t] = false;
            s0[t] = rects[t].area();
            wid0[t] = rects[t].width;
            history.append(t, boxes[t]);
        }
    }

    /* Global retrieval  */
    void HiT::match(cv::Mat& image, std::vector<Box>& searches, bool* mask)
    {
        int objects = searches.size();
        cv::Mat img, squ(cv::Size(1, 1), CV_32FC1);
        for(int n = 0; n < targets; n++) if(*(mask + n))
        {
            int match;
            float square, difference = 1;
            for(int rect = 0; rect < objects; rect++)
            {
                cv::resize(image(searches[rect]), img, templates[n].size());
                cv::matchTemplate(img, templates[n], squ, cv::TM_SQDIFF_NORMED);
                square = squ.at<float>(0, 0);
                if(square < difference)
                {
                    bool matched = true;
                    for(int t = 0; t < targets; t++)
                    {
                        if(t == n) continue;
                        if((searches[rect] & boxes[t]).area() > 0)
                        {
                            matched = false; break;
                        }
                    }
                    if(matched)
                    {
                        match = rect;
                        difference = square;
                    }
                }
            }
            if(retrieval[n] = (difference < retrieval_threshold))
                boxes[n] = searches[match];
        }
    }

    /* Decode results */
    bool HiT::decode(const int n)
    {
        int index = 0;
        float s, score = 0;
        bool success = false;

        /* MLP score */
        for(int i = 0; i < SZ2; i++)
            if(mlp[n][i] > mlp_threshold)
            {
                success = true; break;
            }

        /* Find maximum score */
        for(int i = 0; i < SZ2; i++)
            if((s = weights[i] * scores[n][i]) > score)
                score = s, index = i;

        /* Build box */
        predicts[n][2] = sizes[n][index];
        predicts[n][3] = sizes[n][index + SZ2];
        predicts[n][0] = ((index & 0xF) + offsets[n][index]) / SZ;
        predicts[n][1] = ((index >> 04) + offsets[n][index + SZ2]) / SZ;

        return success && score > score_threshold;
    }

    /* Do inference */
    void HiT::inference(float* ratio, const int h, const int w)
    {
        /* Create stream */
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        /* Apply for memory from cuda */
        MALLOC(MLP, SZ, 1);
        MALLOC(SIZE, SZ, 2);
        MALLOC(SCORE, SZ, 1);
        MALLOC(OFFSET, SZ, 2);
        MALLOC(SEARCH, SIZE2, 3);
        MALLOC(TEMPLATE, SIZE1, 3);
        
        for(int t = 0; t < targets; t++)
        {
            /* Copy data to cuda */
            CPU2CUDA(SEARCH, search[t], SIZE2, 3);
            CPU2CUDA(TEMPLATE, temp[t], SIZE1, 3);

            /* Start inference */
            for(int i = 0; i < M_IDX; i++)
                context->setTensorAddress(NAME[i], buffers[indexes[i]]);
            context->enqueueV3(stream);

            /* Copy results to cpu */
            CUDA2CPU(MLP, mlp[t], SZ, 1);
            CUDA2CPU(SIZE, sizes[t], SZ, 2);
            CUDA2CPU(SCORE, scores[t], SZ, 1);
            CUDA2CPU(OFFSET, offsets[t], SZ, 2);

            /* Post-processing in advance */
            if(t) post_process(ratio, h, w, t - 1);

            /* Wait for inference to complete */
            cudaStreamSynchronize(stream);
        }

        /* Release memory and stream */
        for(int i = 0; i < M_IDX; i++)
            cudaFree(buffers[indexes[i]]);
        cudaStreamDestroy(stream);
    }

    /* Take the area of interest from image and resize it */
    void HiT::sample(cv::Mat& image, cv::Mat* images, int mode, float* resize)
    {
        using namespace std;
        const int height = image.rows, width = image.cols;
        for(int t = 0; t < targets; t++)
        {
            /* Get template (mode = 0) or search space (mode = 1) */
            float x = boxes[t].x, y = boxes[t].y;
            float w = boxes[t].width, h = boxes[t].height;
            float crop = ceil(sqrt(w * h) * (mode + 1) * 2); int
            x1 = round(x + (w - crop) / 2), y1 = round(y + (h - crop) / 2),
            x2 = round(x + (w + crop) / 2), y2 = round(y + (h + crop) / 2);

            /* Get target area */
            int down = max(y2 - height + 1, 0), right = max(x2 - width + 1, 0),
            up = max(0, -y1), left = max(0, -x1), size = (mode? SIZE2 : SIZE1);
            images[t] = image(cv::Rect(
                x1 + left, y1 + up, x2 - x1 - right - left, y2 - y1 - up - down
            ));

            /* Proprotion-preserving resize */
            cv::copyMakeBorder(
                images[t], images[t], up, down, left, right, cv::BORDER_CONSTANT
            );
            cv::resize(images[t], images[t], cv::Size(size, size));

            /* Record the resize ratio */
            if(mode) resize[t] = size / crop;
        }
    }

    /* Preprocess: sample & normalization */
    void HiT::preprocess(cv::Mat& image, bool infer, float** array, float* ptr)
    {
        const int size = infer? SIZE2: SIZE1;
        const int size2 = size * size;
        const int sz = 3 * size2;

        /* Get target area */
        cv::Mat images[N];
        sample(image, images, infer, ptr);

        /* Normalization */
        for(int t = 0; t < targets; t++)
            for(int y = 0; y < size; y++)
                for(int x = 0; x < size; x++)
                    for(int z = 0; z < 3; z++)
                        *((float*)array + t * sz + z * size2 + y * size + x) = (
                            images[t].at<cv::Vec3b>(y, x)[z] / 255.0 - MEAN[z]
                        ) / STD[z];
    }

    /* Post process: remapping, clip & motion tracking */
    void HiT::post_process(float* size, const int h0, const int w0, const int n)
    {
        /* Build box */
        if(!(lost[n] = !decode(n)))
        {
            last[n] = now() * 1e-6;
            if(retrieval[n])
            {
                history.clear(n);
                motion[n] = false;
            }
        }

        /* Coordinate remapping */
        if(lost[n] && retrieval[n])
        {
            motion[n] = true;
            boxes[n] = history.end(n);
        }
        else
        {
            double scale = 1 / size[n];
            double half = (SIZE2 >> 1) * scale,
            x = predicts[n][0] * SIZE2 * scale,
            y = predicts[n][1] * SIZE2 * scale,
            w = predicts[n][2] * SIZE2 * scale,
            h = predicts[n][3] * SIZE2 * scale,
            x0 = boxes[n].x + boxes[n].width / 2,
            y0 = boxes[n].y + boxes[n].height / 2;
            double x1 = x + x0 - half - w / 2, y1 = y + y0 - half - h / 2;
            double x2 = x1 + w, y2 = y1 + h; w = w0; h = h0; CLIP;
            
            /* Motion track */
            if(motion[n] || lost[n])
            {
                float s = boxes[n].area();
                if((motion[n] = !(
                    area * boxes[n].width > wid0[n] &&
                    area * wid0[n] > boxes[n].width &&
                    area * s > s0[n] && area * s0[n] > s
                ))){
                    Box rect = history.end(n);
                    Box rect0 = history.begin(n);
                    boxes[n].width = rect.width * (alpha + 1);
                    boxes[n].height = rect.height * (beta + 1);
                    y1 = boxes[n].y = rect.y - rect.height * beta / 2;
                    x1 = boxes[n].x = rect.x - rect.width * alpha / 2 + (
                        rect.x - rect0.x + (rect.width - rect0.width) * 0.5
                    ) / history.len[n];

                    /* Clip Box */
                    x2 = x1 + boxes[n].width, y2 = y1 + boxes[n].height; CLIP;
                }
            }
        }

        /* Update history */
        retrieval[n] = false;
        history.append(n, boxes[n]);
    }

    /* Resize arrays */
    void BoxArrays::resize(int size)
    {
        length = size;
        for(int n = 0; n < N; n++)
        {
            len[n] = 0;
            pointers[n] = -1;
            arrays[n].resize(length);
        }
    }

    /* Clear a list */
    void BoxArrays::clear(int n)
    {
        len[n] = 0;
        pointers[n] = -1;
    }

    /* Get the last element */
    Box BoxArrays::end(int n)
    {
        return arrays[n][pointers[n]];
    }

    /* Get the first element */
    Box BoxArrays::begin(int n)
    {
        if(len[n] == length && pointers[n] + 1 < length)
            return arrays[n][pointers[n] + 1];
        return arrays[n][0];
    }
 
    /* Add an element */
    void BoxArrays::append(int n, cv::Rect rect)
    {
        pointers[n] = (pointers[n] + 1) % length;
        len[n] = std::min(len[n] + 1, length);
        arrays[n][pointers[n]] = rect;
    }
}
