#include "rtmpose.h"

#include <iostream>
#include <thread>


static cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width, float scale_height, int output_image_width, int output_image_height, bool inverse = false)
{
	// solve the affine transformation matrix

	// get the three points corresponding to the source picture and the target picture
	cv::Point2f src_point_1;
	src_point_1.x = center_x;
	src_point_1.y = center_y;

	cv::Point2f src_point_2;
	src_point_2.x = center_x;
	src_point_2.y = center_y - scale_width * 0.5;

	cv::Point2f src_point_3;
	src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
	src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);


	float alphapose_image_center_x = output_image_width / 2;
	float alphapose_image_center_y = output_image_height / 2;

	cv::Point2f dst_point_1;
	dst_point_1.x = alphapose_image_center_x;
	dst_point_1.y = alphapose_image_center_y;

	cv::Point2f dst_point_2;
	dst_point_2.x = alphapose_image_center_x;
	dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

	cv::Point2f dst_point_3;
	dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
	dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);


	cv::Point2f srcPoints[3];
	srcPoints[0] = src_point_1;
	srcPoints[1] = src_point_2;
	srcPoints[2] = src_point_3;

	cv::Point2f dstPoints[3];
	dstPoints[0] = dst_point_1;
	dstPoints[1] = dst_point_2;
	dstPoints[2] = dst_point_3;

	// get affine matrix
	cv::Mat affineTransform;
	if (inverse)
	{
		affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
	}
	else
	{
		affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
	}

	return affineTransform;
}

RTMPose::RTMPose()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int RTMPose::load(const char *modeltype, const int *target_size, const float *_mean_vals, const float *_norm_vals, bool use_gpu)
{
    rtmpose.clear();
    // blob_pool_allocator.clear();
    // workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    rtmpose.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    rtmpose.opt.num_threads = ncnn::get_big_cpu_count();
    // rtmpose.opt.blob_allocator = &blob_pool_allocator;
    // rtmpose.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "../models/%s.param", modeltype);
    sprintf(modelpath, "../models/%s.bin", modeltype);

    rtmpose.load_param(parampath);
    rtmpose.load_model(modelpath);

    net_w = target_size[0];
    net_h = target_size[1];
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

std::pair<cv::Mat, cv::Mat> RTMPose::preprocess(const cv::Mat& input_image, const cv::Rect &box){
    std::pair<cv::Mat, cv::Mat> result_pair;

	if (!input_image.data)
	{
		return result_pair;
	}

	// deep copy
	cv::Mat input_mat_copy;
	input_image.copyTo(input_mat_copy);

	// calculate the width, height and center points of the human detection box
	int box_width = box.width;
	int box_height = box.height;
	int box_center_x = box.x + box_width / 2;
	int box_center_y = box.y + box_height / 2;

	float aspect_ratio = (float)net_w / net_h;

	// adjust the width and height ratio of the size of the picture in the RTMPOSE input
	if (box_width > (aspect_ratio * box_height))
	{
		box_height = box_width / aspect_ratio;
	}
	else if (box_width < (aspect_ratio * box_height))
	{
		box_width = box_height * aspect_ratio;
	}

	float scale_image_width = box_width * 1.25;
	float scale_image_height = box_height * 1.25;

	// get the affine matrix
	cv::Mat affine_transform = GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		192,
		256
	);

	cv::Mat affine_transform_reverse = GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		192,
		256,
		true
	);

	// affine transform
	cv::Mat affine_image;
	cv::warpAffine(input_mat_copy, affine_image, affine_transform, cv::Size(192, 256), cv::INTER_LINEAR);
	// cv::imwrite("affine_img.jpg", affine_image);

	result_pair = std::make_pair(affine_image, affine_transform_reverse);

	return result_pair;
}

void RTMPose::detect_pose(const cv::Mat &bgr, cv::Rect &bbox, std::vector<keypoint> &points)
{   
    int w = bgr.cols;
    int h = bgr.rows;
    // cv::Rect bbox(0, 0, w, h);
    std::pair<cv::Mat, cv::Mat> crop_result_pair = preprocess(bgr, bbox); // crop_mat and affine_transform_reverse

    cv::Mat crop_mat = crop_result_pair.first;
	cv::Mat affine_transform_reverse = crop_result_pair.second;
    // std::cout << crop_mat.size << std::endl;

    ncnn::Mat in = ncnn::Mat::from_pixels(crop_mat.data, ncnn::Mat::PIXEL_BGR, net_w, net_h);

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = rtmpose.create_extractor();
    // std::cout << in.c << " " << in.h << " " << in.w << std::endl;
    ex.input("in0", in);

    ncnn::Mat simcc_x, simcc_y;
    ex.extract("out0", simcc_x);
    ex.extract("out1", simcc_y);

	int simcc_x_width = simcc_x.w;
	int simcc_y_height = simcc_y.w;

	float* simcc_x_result = (float*) simcc_x.data;
	float* simcc_y_result = (float*) simcc_y.data;


	for (int i = 0; i < num_joints; ++i)
	{
		// find the maximum and maximum indexes in the value of each Extend_width length
		auto x_biggest_iter = std::max_element(simcc_x_result + i * simcc_x_width, simcc_x_result + i * simcc_x_width + simcc_x_width);
		int max_x_pos = std::distance(simcc_x_result + i * simcc_x_width, x_biggest_iter);
		int pose_x = max_x_pos / 2;
		float score_x = *x_biggest_iter;

		// find the maximum and maximum indexes in the value of each exten_height length
		auto y_biggest_iter = std::max_element(simcc_y_result + i * simcc_y_height, simcc_y_result + i * simcc_y_height + simcc_y_height);
		int max_y_pos = std::distance(simcc_y_result + i * simcc_y_height, y_biggest_iter);
		int pose_y = max_y_pos / 2;
		float score_y = *y_biggest_iter;

		//float score = (score_x + score_y) / 2;
		float score = std::max(score_x, score_y);

		keypoint temp_point;
		temp_point.x = int(pose_x);
		temp_point.y = int(pose_y);
		temp_point.score = score;
        // std::cout << score << std::endl;
		points.emplace_back(temp_point);
	}

	// anti affine transformation to obtain the coordinates on the original picture
	for (int i = 0; i < points.size(); ++i)
	{
		cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
		origin_point_Mat.at<double>(0, 0) = points[i].x;
		origin_point_Mat.at<double>(1, 0) = points[i].y;

		cv::Mat temp_result_mat = affine_transform_reverse * origin_point_Mat;

		points[i].x = temp_result_mat.at<double>(0, 0);
		points[i].y = temp_result_mat.at<double>(1, 0);
	}

        // filter joint jitter
    for (int i = 0; i < num_joints; ++i)
    {
        points[i].y = joint_filters.filters[i].first.filter(points[i].y);
        points[i].x = joint_filters.filters[i].second.filter(points[i].x);
    }
}

int RTMPose::draw(cv::Mat &bgr, std::vector<keypoint> &points)
{
    int skele_index[][2] = {{0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {11, 12}, {5, 11}, {11, 13}, {13, 15}, {6, 12}, {12, 14}, {14, 16}};
    int color_index[][3] = {
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 0, 255},
    };
    // cv::resize(bgr, bgr, cv::Size(192, 192));

    for (int i = 0; i < num_joints; i++)
    {
        if (points[skele_index[i][0]].score > 0.1 && points[skele_index[i][1]].score > 0.1)
            cv::line(bgr, cv::Point(points[skele_index[i][0]].x, points[skele_index[i][0]].y),
                     cv::Point(points[skele_index[i][1]].x, points[skele_index[i][1]].y), cv::Scalar(color_index[i][0], color_index[i][1], color_index[i][2]), 2);
    }
    for (int i = 0; i < num_joints; i++)
    {
        if (points[i].score > 0.1)
            cv::circle(bgr, cv::Point(points[i].x, points[i].y), 3, cv::Scalar(100, 255, 150), -1);
    }
    
    return 0;
}