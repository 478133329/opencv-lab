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

// threshold(Mat& src, double th, double val1, double val2, Mat& dst);
Mat bin_val(Mat& src, int th);

Mat log_trans(Mat& src, int c);

Mat gamma_trans(Mat& src, int c, double gamma);

Mat complement_trans(Mat& src);