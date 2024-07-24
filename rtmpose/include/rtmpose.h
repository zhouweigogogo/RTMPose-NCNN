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

#ifndef RTMPOSE_H
#define RTMPOSE_H

#include <opencv2/core/core.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

#include "jitterfilter.h"
#include <net.h>
#include <cpu.h>
#include "rtmdet.h"

using namespace std;
struct keypoint
{
    float x;
    float y;
    float score;
};

const int num_joints = 17;

class RTMPose
{
public:
    RTMPose();

    int load(const char *modeltype, const int *target_size, const float *mean_vals, const float *norm_vals, bool use_gpu = false);

    std::pair<cv::Mat, cv::Mat> preprocess(const cv::Mat& input_image, const cv::Rect &box);

    void detect_pose(const cv::Mat &bgr, cv::Rect &bbox, std::vector<keypoint> &points);

    int draw(cv::Mat &bgr, std::vector<keypoint> &points);

private:
    ncnn::Net rtmpose;
    int feature_size;
    float kpt_scale;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    int net_w;
    int net_h;
    std::vector<std::vector<float>> dist_y, dist_x;

    OneEuroFilterJoints joint_filters;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // MOVENET_H
