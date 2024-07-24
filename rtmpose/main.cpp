#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "rtmdet.h"
#include "rtmpose.h"
#include <benchmark.h>

static int draw_unsupported(cv::Mat &rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat &rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

int image_demo(RTMDet &rtmdet, RTMPose &rtmpose, const char *imagepath)
{
    std::vector<Object> objects;
    std::vector<keypoint> points;
    cv::Mat image = cv::imread(imagepath);
    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    int64 pre_start = cv::getTickCount();
    rtmdet.detect(image, objects);
    // rtmdet.draw(image, objects);

    cv::Rect bbox = objects[0].rect;
    rtmpose.detect_pose(image, bbox, points);
    rtmpose.draw(image, points);

    int64 pre_end = cv::getTickCount();
    double pre_time = (pre_end - pre_start) / cv::getTickFrequency();

    std::cout << "cost time: " << pre_time << " s" << std::endl;
    objects.clear();
    points.clear();
    cv::imwrite("../output/result.png", image);
    return 0;
}
int webcam_demo(RTMDet &rtmdet, RTMPose &rtmpose, int cam_id)
{
    cv::Mat bgr;
    cv::VideoCapture cap(cam_id, cv::CAP_V4L2);
    std::vector<keypoint> points;
    std::vector<Object> objects;

    int image_id = 0;
    while (true)
    {
        cap >> bgr;

        rtmdet.detect(bgr, objects);

        for (int i = 0; i < objects.size(); i++)
        {
            cv::Rect bbox = objects[i].rect;
            rtmpose.detect_pose(bgr, bbox, points);
            rtmpose.draw(bgr, points);
            points.clear();
        }
        objects.clear();
        draw_fps(bgr);
        cv::imshow("test", bgr);
        cv::waitKey(1);
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        return -1;
    }
    const char *modeltypes[] =
        {
            "rtmdet-nano",
            "rtmpose-t"
        };

    const int target_sizes[][2] =
        {
            {320, 320},
            {192, 256}, // w h
        };

    const float mean_vals[][3] =
        {
            {103.53f, 116.28f, 123.675f},
            {123.675f, 116.28f, 103.53f},
        };

    const float norm_vals[][3] =
        {
            {1 / 57.375f, 1 / 57.12f, 1 / 58.395f}, // BGR
            {1 / 58.395f, 1 / 57.12f, 1 / 57.375f}, // RGB
        };

    int mode = atoi(argv[1]);
    switch (mode)
    {
    case 0:
    {
        int cam_id = atoi(argv[2]);

        const char *images = argv[2];
        RTMDet rtmdet;
        int det_id = 0;
        const char *det_type = modeltypes[(int)det_id];
        bool use_gpu = false;
        rtmdet.load(det_type, target_sizes[(int)det_id], mean_vals[(int)det_id], norm_vals[(int)det_id], use_gpu);
        std::cout << "RTMDet load successfully!" << std::endl;

        RTMPose rtmpose;
        int pose_id = 1;
        const char *pose_type = modeltypes[(int)pose_id];
        rtmpose.load(pose_type, target_sizes[(int)pose_id], mean_vals[(int)pose_id], norm_vals[(int)pose_id], use_gpu);
        std::cout << "RTMPose load successfully!" << std::endl;

        webcam_demo(rtmdet, rtmpose, cam_id);
    }
    case 1:
    {
        const char *images = argv[2];
        RTMDet rtmdet;
        int det_id = 0;
        const char *det_type = modeltypes[(int)det_id];
        bool use_gpu = false;
        rtmdet.load(det_type, target_sizes[(int)det_id], mean_vals[(int)det_id], norm_vals[(int)det_id], use_gpu);
        std::cout << "RTMDet load successfully!" << std::endl;

        RTMPose rtmpose;
        int pose_id = 1;
        const char *pose_type = modeltypes[(int)pose_id];
        rtmpose.load(pose_type, target_sizes[(int)pose_id], mean_vals[(int)pose_id], norm_vals[(int)pose_id], use_gpu);
        std::cout << "RTMPose load successfully!" << std::endl;

        image_demo(rtmdet, rtmpose, images);
        break;
    }

    default:
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n", argv[0]);
        break;
    }
    }
}
