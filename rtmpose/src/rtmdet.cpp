// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "rtmdet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

static float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(objects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
            {
                continue;
            }

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
            {
                keep = 0;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat &feat_blob, int stride, float prob_threshold, std::vector<Object> &objects)
{
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    // const float prob_threshold_prev = -log((1 - prob_threshold) / (prob_threshold + 1e-5));
    // std::cout << num_grid_y << " " << num_grid_x << " " << num_w << std::endl;
    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const float *matat = feat_blob.channel(i).row(j);
            float score = matat[0];
            if (score >= prob_threshold)
            {
                score = sigmoid(score);
                // std::cout << j << " " << i << " " << score << " " << matat[1] << " " << matat[2] << " " << matat[3] << " " << matat[4] << std::endl;
                float x0 = j * stride - matat[1];
                float y0 = i * stride - matat[2];
                float x1 = j * stride + matat[3];
                float y1 = i * stride + matat[4];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = 0;
                obj.prob = score;
                objects.push_back(obj);
            }
        }
    }
}

RTMDet::RTMDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int RTMDet::load(const char *modeltype, const int *target_size, const float *_mean_vals, const float *_norm_vals, bool use_gpu)
{
    rtmdet.clear();
    // blob_pool_allocator.clear();
    // workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    rtmdet.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    rtmdet.opt.num_threads = ncnn::get_big_cpu_count();
    // rtmdet.opt.blob_allocator = &blob_pool_allocator;
    // rtmdet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "../models/%s.param", modeltype);
    sprintf(modelpath, "../models/%s.bin", modeltype);

    rtmdet.load_param(parampath);
    rtmdet.load_model(modelpath);

    net_h = target_size[0];
    net_w = target_size[1];
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int RTMDet::detect(const cv::Mat &bgr, std::vector<Object> &objects, float prob_threshold, float nms_threshold)
{

    int width = bgr.cols;
    int height = bgr.rows;

    int w = width;
    int h = height;
    float ratio_w = (float)width / net_w;
    float ratio_h = (float)height / net_h;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, width, height, net_w, net_h);

    int wpad = net_w - w;
    int hpad = net_h - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = rtmdet.create_extractor();
    std::cout << in_pad.c << " " << in_pad.h << " " << in_pad.w << std::endl;
    ex.input("images", in);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat out0;

        ex.extract("1211", out0);

        std::vector<Object> objects8;
        generate_proposals(out0, 8, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out1;
        ex.extract("1213", out1);

        std::vector<Object> objects16;
        generate_proposals(out1, 16, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out2;
        ex.extract("1215", out2);

        std::vector<Object> objects32;
        generate_proposals(out2, 32, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold, true);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = objects[i].rect.x * ratio_w;
        float y0 = objects[i].rect.y * ratio_h;
        float x1 = (objects[i].rect.x + objects[i].rect.width) * ratio_w;
        float y1 = (objects[i].rect.y + objects[i].rect.height) * ratio_h;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

int RTMDet::draw(cv::Mat &bgr, std::vector<Object> &objects)
{

    static const unsigned char colors[2][3] = {
        {0, 255, 0},
        {0, 0, 255},
    };
    for (int i = 0; i < objects.size(); i++)
    {
        const Object &obj = objects[i];
        // std::cout << obj.rect << std::endl;

        //         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char *color = colors[obj.label];

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(bgr, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }

    return 0;
}