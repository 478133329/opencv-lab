#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void get_mean_smoothing_img(const Mat& src_img, Mat& dest_img, Size ksize);

void get_gauss_mask(Mat& mask, Size ksize, double sigma);

void get_gauss_smoothing_img(const Mat& src_img, Mat& dest_img, Mat mask);

void get_enhance_filter_img(const Mat& src, Mat& dst, Size ksize, double k);

void get_laplacian_img(const Mat& src, Mat& dst);

void get_rob_sob_img(const Mat& src, Mat& dst, int x[][3], int y[][3]);