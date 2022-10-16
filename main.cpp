#include <vector>
#include <unordered_map>
#include "hist.h"
#include "gray.h"
#include "filtering.h"

void showImgsAndHistImgs(vector<Mat*> vec)
{
	int img_num = vec.size();
	for (int i = 0; i < img_num; i++)
	{
		resize(*vec[i], *vec[i], Size(300, 300));
	}
	Size conSize(300 + 256 * 3, 300 * img_num);
	Mat dst(conSize, CV_8UC1);
	int py = 0;
	for (int i = 0; i < img_num; i++)
	{
		Mat hist = get_hist_img(*vec[i], 256);
		resize(hist, hist, Size(256 * 3, 300), INTER_LINEAR);
		for (int row = 0; row < vec[i]->rows; row++)
		{
			for (int col = 0; col < vec[i]->cols; col++)
			{
				dst.at<uchar>(py + row, col) = vec[i]->at<uchar>(row, col);
			}
		}
		for (int row = 0; row < hist.rows; row++)
		{
			for (int col = 0; col < hist.cols; col++)
			{
				dst.at<uchar>(py + row, 300 + col) = hist.at<uchar>(row, col);
			}
		}
		py = 300 * (i + 1);
	}
	namedWindow("对比", WINDOW_NORMAL);
	imshow("对比", dst);
}

// 二值变换
void demo1()
{
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, Size(1280, 960));
	cvtColor(src_img, src_img, COLOR_RGB2GRAY);
	Mat bin_img = bin_val(src_img, 128);
	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", src_img);
	namedWindow("黑白图", WINDOW_NORMAL);
	imshow("黑白图", bin_img);
}

// 灰度变换（对数变换和伽马变换）
void demo2()
{
	Mat src_img = imread("qingdao.jpg");
	resize(src_img, src_img, Size(1280, 960));
	cvtColor(src_img, src_img, COLOR_RGB2GRAY);
	Mat log_img = log_trans(src_img, 10);
	Mat gamma_img = gamma_trans(src_img, 1.5, 0.8);
	vector<Mat*> images;                // vector可以存放指针，不能存放引用。
	images.push_back(&src_img);
	images.push_back(&log_img);
	images.push_back(&gamma_img);
	showImgsAndHistImgs(images);
}

// 补色变换
void demo3()
{
	Mat src_img = imread("qingdao.jpg");
	resize(src_img, src_img, Size(1280, 960));
	Mat complement_img = complement_trans(src_img);
	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", src_img);
	namedWindow("补色图像", WINDOW_NORMAL);
	imshow("补色图像", complement_img);
}

// 原图的彩色均衡图和灰度均衡图
void demo4()
{
	int bins = 256;
	Size size(300, 300);
	Mat src_img = imread("qingdao.jpg");
	resize(src_img, src_img, size);
	namedWindow("原图", WINDOW_AUTOSIZE);
	imshow("原图", src_img);

	Mat dst = get_equalized_rgb_img(src_img, 256);
	namedWindow("原图均衡图", WINDOW_AUTOSIZE);
	imshow("原图均衡图", dst);

	cvtColor(src_img, src_img, COLOR_RGB2GRAY);
	namedWindow("原图灰度图", WINDOW_AUTOSIZE);
	imshow("原图灰度图", src_img);

	Mat equal_src_img = get_equalized_gray_img(src_img, bins);
	Mat equal_src_img_hist_img = get_hist_img(equal_src_img, bins);
	namedWindow("原图灰度图均衡化", WINDOW_AUTOSIZE);
	imshow("原图灰度图均衡化", equal_src_img);

}

// 原图规定化
void demo5()
{
	int bins = 256;
	Size size(1280, 960);
	Mat src_img = imread("qingdao.jpg");
	resize(src_img, src_img, size);
	cvtColor(src_img, src_img, COLOR_RGB2GRAY);

	namedWindow("原图灰度图", WINDOW_NORMAL);
	imshow("原图灰度图", src_img);

	Mat src_img_hist_img = get_hist_img(src_img, bins);
	namedWindow("原图灰度图直方图", WINDOW_NORMAL);
	imshow("原图灰度图直方图", src_img_hist_img);

	Mat target_img = imread("background.jpg");
	resize(target_img, target_img, size);
	cvtColor(target_img, target_img, COLOR_RGB2GRAY);
	Mat target_img_hist_img = get_hist_img(target_img, bins);
	namedWindow("目标图直方图", WINDOW_NORMAL);
	imshow("目标图直方图", target_img_hist_img);

	Mat matched_img = get_matched_img(src_img, target_img);
	namedWindow("规定化原图灰度图", WINDOW_NORMAL);
	imshow("规定化原图灰度图", matched_img);

	Mat matched_img_hist_img = get_hist_img(matched_img, bins);
	namedWindow("规定化原图灰度图直方图", WINDOW_NORMAL);
	imshow("规定化原图灰度图直方图", matched_img_hist_img);
}

// 图像平滑处理
void demo6() {
	Mat src_img = imread("img/qingdao.jpg");
	Mat mean_smoothing_img = Mat::zeros(src_img.size(), src_img.type());
	get_mean_smoothing_img(src_img, mean_smoothing_img, Size(3, 3));
	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", src_img);
	namedWindow("均值平滑处理图", WINDOW_NORMAL);
	imshow("均值平滑处理图", mean_smoothing_img);
}

// 彩色图像平滑处理
void demo7() {
	Mat src_img = imread("img/qingdao.jpg");
	Mat mask, gauss_smoothing_img;
	get_gauss_mask(mask, Size(5, 5), 0.8);
	get_gauss_smoothing_img(src_img, gauss_smoothing_img, mask);
	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", src_img);
	namedWindow("高斯平滑处理图", WINDOW_NORMAL);
	imshow("高斯平滑处理图", gauss_smoothing_img);
}

int main()
{
	cout << "1:二值变换" << endl << "2:灰度变换" << endl << "3:补色变换" << endl << "4:彩色图和灰度图的直方图均衡化" << endl << "5:灰度图的直方图规定化" << endl << "6:灰度图像平滑处理" << endl
		<< "7:彩色图像平滑处理" << endl;
	int flag;
	cin >> flag;
	switch (flag)
	{
	case 1:
		// 滑块修改阈值、两个图片展示在一个框。
		demo1();
		break;
	case 2:
		// 命令行输入参数，汉字显示？
		demo2();
		break;
	case 3:
		// 合成一张图
		demo3();
		break;
	case 4:
		// 合成一张图，并展示灰度图和彩色图的直方图分布
		demo4();
		break;
	case 5:
		// 合成一张图
		demo5();
		break;
	case 6:
		// 灰度图像平滑处理
		demo6();
		break;
	case 7:
		// 彩色图像平滑处理
		demo7();
		break;
	case 99:
		// test
	{	

	}
		break;
	default:
		cout << "input error." << endl;
	}
	waitKey(0);
	return 0;
}