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
// #define _CRT_SECURE_NO_WARNINGS
// #define NOMINMAX
#ifndef RTMDET_H
#define RTMDET_H

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <net.h>

#define MAX_STRIDE 32

using namespace std;
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class RTMDet
{
public:
    RTMDet();

    int load(const char *modeltype, const int *target_size, const float *mean_vals, const float *norm_vals, bool use_gpu = false);

    int detect(const cv::Mat &bgr, std::vector<Object> &objects, float prob_threshold = 0.65f, float nms_threshold = 0.65f);

    int draw(cv::Mat &bgr, std::vector<Object> &objects);

private:
    ncnn::Net rtmdet;
    int net_h;
    int net_w;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // NANODET_H
#pragma once