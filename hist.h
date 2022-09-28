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

// ��ȡ�Ҷ�����
Mat get_hist(const Mat& mat);

// ��ȡ�Ҷ�����ֱ��ͼ
Mat get_hist_img(Mat& mat, int bins);

// ��ȡֱ��ͼ���⻯ͼ��
Mat get_equalized_gray_img(Mat& src_img, int bins);

// ��ȡֱ��ͼ���⻯ͼ��
Mat get_equalized_rgb_img(Mat& src_img, int bins);

// ��ȡ���⻯�ĻҶ�ӳ���
unordered_map<int, int> get_equal_map(Mat hist);

// ��ȡ��ӳ���
unordered_map<int, int> reverse_map(unordered_map<int, int> m);

// ��ȡֱ��ͼ�涨��ͼ��
Mat get_matched_img(Mat& src_img, const Mat& target_img);