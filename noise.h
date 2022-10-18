#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat salt_pepper_noise(const Mat& input, int n, int value);

Mat gauss_noise(const Mat& input, int avg, int sd);

Mat filter_process(const Mat& img, uchar kernel(const Mat&, int, int, int, int), int size, int q = 1);

// 避免在头文件中声明static函数
uchar arithmetic_mean_kernel(const Mat& img, int size, int i, int j, int q);

uchar geometric_mean_kernel(const Mat& img, int size, int i, int j, int q);

uchar harmonic_mean_kernel(const Mat& img, int size, int i, int j, int q);

uchar antiharmonic_mean_kernel(const Mat& img, int size, int i, int j, int q);

int quick_select(vector<uchar>& vec, int left, int right, int target);

uchar median_kernel(const Mat& img, int size, int i, int j, int q);

uchar adaptive_mean_kernel(const Mat& img, int size, int i, int j, int q);

uchar adaptive_median_kernel(const Mat& img, int size, int i, int j, int max_size);