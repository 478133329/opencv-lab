#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;
using namespace cv;

// 获取灰度数组
Mat get_hist(const Mat& mat);

// 获取灰度数组直方图
Mat get_hist_img(Mat& mat, int bins);

// 获取直方图均衡化图像
Mat get_equalized_gray_img(Mat& src_img, int bins);

// 获取直方图均衡化图像
Mat get_equalized_rgb_img(Mat& src_img, int bins);

// 获取均衡化的灰度映射表
unordered_map<int, int> get_equal_map(Mat hist);

// 获取逆映射表
unordered_map<int, int> reverse_map(unordered_map<int, int> m);

// 获取直方图规定化图像
Mat get_matched_img(Mat& src_img, const Mat& target_img);