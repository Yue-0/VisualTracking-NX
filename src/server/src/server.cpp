/* @Author: YueLin */

/* For multithreading */
#include <thread>

/* ROS */
#include <ros/ros.h>
#include <std_srvs/SetBool.h>

/* Head files */
#include "sot/HiT.hpp"
#include "mot/YOLO.hpp"
#include "mot/ByteTrack.hpp"

/* Custom services */
#include "tracking_msgs/Image.h"
#include "tracking_msgs/ImageWithBoxes.h"

/* Callback function */
#define callback(f, ns) f(ns::Request& req, ns::Response& res)

/* Create image from request */
#define req2mat(req, mat) cv::Mat mat(\
    req.height, req.width, CV_8UC3, req.data.data()\
)

/* Pointers */
void *transformer, *cnn, *algorithm;

/* Global retrieval function for SOT */
enum Retrieval{DISABLE = 0, SOT_ONLY, ALWAYS} use_retrieval;
void retrieval(cv::Mat& img, sot::HiT* hit, void* detector)
{
    double t = now() * 1e-6;
    bool detection[sot::N + 1] = {false};
    for(int b = 0; b < hit->targets; b++)
        if(detection[b + 1] = hit->lost[b] && t - hit->last[b] > hit->wait)
        {
            hit->last[b] = t;
            detection[0] = true;
        }
    if(detection[0])
    {
        cv::Mat image = img.clone();
        std::vector<sot::Box> objects;
        const int height = image.rows, width = image.cols;
        det::YOLO* yolo = reinterpret_cast<det::YOLO*>(detector);
        for(Object object: yolo->detect(image, hit->retrieval_threshold))
        {
            cv::Point2f br = object.rect.br(), tl = object.rect.tl();
            if(std::min(tl.x, tl.y) >= 0 && br.x < width && br.y < height)
                objects.push_back(object.rect);
        }
        if(!objects.empty())
            hit->match(img, objects, detection + 1);
    }
}

/* Callback function for reseting tracker */
bool callback(reset, std_srvs::SetBool)
{
    mot::ByteTrack* tracker = reinterpret_cast<mot::ByteTrack*>(algorithm);
    tracker->reset(); return true;
}

/* Callback function for SOT */
bool callback(track1, tracking_msgs::Image)
{
    int lost = 0;  // The number of lost targets

    /* Create cv::Mat from request */
    req2mat(req.image, img);

    /* Get HiT model */
    sot::HiT* hit = reinterpret_cast<sot::HiT*>(transformer);

    /* Inference and speed testing */
    int64 time = now();
    hit->track(img);
    if(use_retrieval) retrieval(img, hit, cnn);
    res.time = (now() - time) * 1e-6;
    
    /* Convert the results into ROS message */
    for(int b = 0; b < hit->targets; b++)
    {
        if(hit->lost[b]) lost++;
        cv::Rect box = hit->boxes[b];
        res.cls.push_back(hit->lost[b]);
        res.h.push_back(box.height);
        res.w.push_back(box.width);
        res.y.push_back(box.y);
        res.x.push_back(box.x);
    }
    ROS_INFO(
        "[SOT] Successfully tracked %d %s, lost %d %s, took %.3fs.",
        hit->targets - lost, hit->targets > lost + 1? "targets": "target",
        lost, lost > 1? "targets": "target", res.time
    );
    return true;
}

/* Callback function for SOT template initialization */
bool callback(init, tracking_msgs::ImageWithBoxes)
{
    /* Create cv::Mat from request */
    req2mat(req.image, img);

    /* Get HiT model */
    sot::HiT* hit = reinterpret_cast<sot::HiT*>(transformer);
    
    /* Create initial boxes */
    cv::Rect boxes[req.num];
    for(int n = 0; n < req.num; n++)
        boxes[n] = cv::Rect(req.x[n], req.y[n], req.w[n], req.h[n]);

    /* Template initialization and speed testing */
    int64 time = now();
    hit->init(img, boxes, req.num);
    res.time = (now() - time) * 1e-6;
    ROS_INFO(
        "[SOT] Initialization successful, total have %d %s, took %.3fs.",
        req.num, req.num > 1? "targets": "target", res.time
    );
    return true;
}

/* Callback function for MOT */
bool callback(track, tracking_msgs::Image)
{
    /* Create cv::Mat from request */
    req2mat(req.image, img);

    /* Get YOLO model and ByteTrack algorithm */
    det::YOLO* yolo = reinterpret_cast<det::YOLO*>(cnn);
    mot::ByteTrack* tracker = reinterpret_cast<mot::ByteTrack*>(algorithm);

    /* Detection, tracking and speed testing */
    int64 time = now();
    Objects objects = yolo->detect(img);
    mot::Boxes boxes = tracker->track(objects);
    mot::Array labels = tracker->match(objects, boxes);
    res.time = (now() - time) * 1e-6;

    /* Convert the results into ROS message */
    const int num = boxes.size();
    for(int b = 0; b < num; b++)
    {
        mot::Box box = boxes[b];
        res.cls.push_back(labels[b]);
        res.id.push_back(box.track_id);
        res.x.push_back(std::round(box.tlwh[0]));
        res.y.push_back(std::round(box.tlwh[1]));
        res.w.push_back(std::round(box.tlwh[2]));
        res.h.push_back(std::round(box.tlwh[3]));
    }
    ROS_INFO(
        "[MOT] Detected %d %s, took %.3fs.",
        num, num > 1? "objects": "object", res.time
    );
    return true;
}

/* Callback function for DET */
bool callback(detect, tracking_msgs::Image)
{
    /* Create cv::Mat from request */
    req2mat(req.image, img);

    /* Get YOLO model and ByteTrack algorithm */
    det::YOLO* yolo = reinterpret_cast<det::YOLO*>(cnn);

    /* Detection and speed testing */
    int64 time = now();
    Objects objects = yolo->detect(img);
    res.time = (now() - time) * 1e-6;

    /* Convert the results into ROS message */
    const int num = objects.size();
    for(int b = num; b;)
    {
        Object object = objects[--b];
        res.cls.push_back(object.label);
        res.x.push_back(std::round(object.rect.x));
        res.y.push_back(std::round(object.rect.y));
        res.w.push_back(std::round(object.rect.width));
        res.h.push_back(std::round(object.rect.height));
    }
    ROS_INFO(
        "[DET] Detected %d %s, took %.3fs.",
        num, num > 1? "objects": "object", res.time
    );
    return true;
}

/* Callback function for SOT&MOT */
bool callback(sot8mot, tracking_msgs::Image)
{
    /* Create cv::Mat from request */
    req2mat(req.image, img);
    cv::Mat img0 = img.clone();

    /* Get YOLO model and ByteTrack algorithm */
    det::YOLO* yolo = reinterpret_cast<det::YOLO*>(cnn);
    mot::ByteTrack* tracker = reinterpret_cast<mot::ByteTrack*>(algorithm);

    /* Enable sub thread for SOT */
    std::thread sot_mot([&]{
        /* Get HiT model */
        sot::HiT* hit = reinterpret_cast<sot::HiT*>(transformer);

        /* Inference */
        int64 time = now();
        hit->track(img0);
        if(use_retrieval == ALWAYS)
            retrieval(img0, hit, yolo);
        res.time = (now() - time) * 1e-6;
        
        /* Convert the results into ROS message */
        res.occ = hit->lost[0];
        res.x0 = hit->boxes[0].x;
        res.y0 = hit->boxes[0].y;
        res.w0 = hit->boxes[0].width;
        res.h0 = hit->boxes[0].height;

        ROS_INFO(
            "[SOT] Successfully tracked %d target, lost %d target, took %.3fs.",
            1 - res.occ, res.occ, res.time
        );
    });

    /* Detection, tracking and speed testing */
    int64 time = now();
    Objects objects = yolo->detect(img);
    mot::Boxes boxes = tracker->track(objects);
    mot::Array labels = tracker->match(objects, boxes);
    res.time = (now() - time) * 1e-6;

    /* Convert the results into ROS message */
    const int num = boxes.size();
    for(int b = 0; b < num; b++)
    {
        mot::Box box = boxes[b];
        res.cls.push_back(labels[b]);
        res.id.push_back(box.track_id);
        res.x.push_back(std::round(box.tlwh[0]));
        res.y.push_back(std::round(box.tlwh[1]));
        res.w.push_back(std::round(box.tlwh[2]));
        res.h.push_back(std::round(box.tlwh[3]));
    }
    ROS_INFO(
        "[MOT] Detected %d %s, took %.3fs.",
        num, num > 1? "objects": "object", res.time
    );
    return sot_mot.join(), true;
}

/* Callback function for SOT&DET */
bool callback(sot8det, tracking_msgs::Image)
{
    /* Create cv::Mat from request */
    req2mat(req.image, img);
    cv::Mat img0 = img.clone();

    /* Get YOLO model */
    det::YOLO* yolo = reinterpret_cast<det::YOLO*>(cnn);

    /* Enable sub thread for SOT */
    std::thread sot_det([&]{
        /* Get HiT model */
        sot::HiT* hit = reinterpret_cast<sot::HiT*>(transformer);

        /* Inference */
        int64 time = now();
        hit->track(img0);
        if(use_retrieval == ALWAYS)
            retrieval(img0, hit, yolo);
        res.time = (now() - time) * 1e-6;
        
        /* Convert the results into ROS message */
        res.occ = hit->lost[0];
        res.x0 = hit->boxes[0].x;
        res.y0 = hit->boxes[0].y;
        res.w0 = hit->boxes[0].width;
        res.h0 = hit->boxes[0].height;

        ROS_INFO(
            "[SOT] Successfully tracked %d target, lost %d target, took %.3fs.",
            1 - res.occ, res.occ, res.time
        );
    });

    /* Detection and speed testing */
    int64 time = now();
    Objects objects = yolo->detect(img);
    res.time = (now() - time) * 1e-6;

    /* Convert the results into ROS message */
    const int num = objects.size();
    for(int b = num; b;)
    {
        Object object = objects[--b];
        res.cls.push_back(object.label);
        res.x.push_back(std::round(object.rect.x));
        res.y.push_back(std::round(object.rect.y));
        res.w.push_back(std::round(object.rect.width));
        res.h.push_back(std::round(object.rect.height));
    }
    ROS_INFO(
        "[DET] Detected %d %s, took %.3fs.",
        num, num > 1? "objects": "object", res.time
    );
    return sot_det.join(), true;
}

int main(int argc, char* argv[])
{
    /* Initialize ROS node */
    ros::init(argc, argv, "server");

    /* Create node handle */
    ros::NodeHandle nh("~");

    /* Initialize HiT model */
    double parameters[] = {
        nh.param("/beta", 1e-3),
        nh.param("/alpha", 2e-3),
        nh.param("/margin", 1e1),
        nh.param("/momentum", 1e2),
        nh.param("/mlp_threshold", 0.6),
        nh.param("/area_threshold", 1.5),
        nh.param("/score_threshold", 0.35),
        nh.param("/retrieve_threshold", 0.8),
        nh.param("/retrieve_wait_time", 1.0)
    };
    ROS_INFO("[SOT] Loading HiT model from %s.", argv[1]);
    sot::HiT hit(argv[1], parameters); transformer = (void*)&hit;
    use_retrieval = (enum Retrieval)nh.param("/use_global_retrieval", 1);

    /* Initialize YOLO model */
    int info[] = {
        nh.param("/infer_size", 480),
        nh.param("/num_classes", 80),
        nh.param("/outputs_dim", 4725)
    };
    double thresholds[] = {
        nh.param("/nms_threshold", 0.45),
        nh.param("/detect_threshold", 0.5)
    };
    ROS_INFO("[DET] Loading YOLO model from %s.", argv[2]);
    det::YOLO yolo(argv[2], thresholds, info); cnn = (void*)&yolo;

    /* Warmup */
    cv::Mat warmup(info[2], info[2], CV_8UC3);
    cv::randu(warmup, cv::Scalar::all(0), cv::Scalar::all(0xFF));
    yolo.detect(warmup); yolo.detect(warmup); warmup.release();

    /* Initialize ByteTrack algorithm */
    ROS_INFO("[MOT] Initialize ByteTrack algorithm.");
    mot::ByteTrack tracker(
        nh.param("/detect_threshold", 0.5),
        nh.param("/match_threshold", 0.8)
    );
    algorithm = (void*)&tracker;

    /* Initialize services */
    std::string services[7];
    nh.getParam("mot_server", services[0]);
    nh.getParam("sot_server", services[1]);
    nh.getParam("det_server", services[2]);
    nh.getParam("box_server", services[3]);
    nh.getParam("reset_server", services[4]);
    nh.getParam("sot_mot_server", services[5]);
    nh.getParam("sot_det_server", services[6]);
    ros::ServiceServer rs = nh.advertiseService(services[4], reset);
    ros::ServiceServer tracking = nh.advertiseService(services[0], track);
    ros::ServiceServer sot_mot = nh.advertiseService(services[5], sot8mot);
    ros::ServiceServer sot_det = nh.advertiseService(services[6], sot8det);
    ros::ServiceServer tracking1 = nh.advertiseService(services[1], track1);
    ros::ServiceServer detection = nh.advertiseService(services[2], detect);
    ros::ServiceServer initialization = nh.advertiseService(services[3], init);

    /* Start servers */
    ROS_INFO("[SOT] Service is ready.");
    ROS_INFO("[DET] Service is ready.");
    ROS_INFO("[MOT] Service is ready.");
    return ros::spin(), 0;
}
