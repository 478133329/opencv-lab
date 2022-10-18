#include <vector>
#include <unordered_map>
#include "hist.h"
#include "gray.h"
#include "filtering.h"
#include "noise.h"

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
	Mat src_img = imread("img/qingdao.jpg");
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
	Mat src_img = imread("img/qingdao.jpg");
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
	Mat src_img = imread("img/qingdao.jpg");
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
	Mat src_img = imread("img/qingdao.jpg");
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

// 平滑处理
void demo6()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat mean_smoothing_img = Mat::zeros(src_img.size(), src_img.type());
	get_mean_smoothing_img(src_img, mean_smoothing_img, Size(5, 5));
	Mat mask, gauss_smoothing_img;
	get_gauss_mask(mask, Size(5, 5), 0.8);
	get_gauss_smoothing_img(src_img, gauss_smoothing_img, mask);
	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", src_img);
	namedWindow("均值平滑处理图", WINDOW_NORMAL);
	imshow("均值平滑处理图", mean_smoothing_img);
	namedWindow("高斯平滑处理图", WINDOW_NORMAL);
	imshow("高斯平滑处理图", gauss_smoothing_img);
}

// 高提升滤波算法
void demo7()
{
	Mat src_img = imread("img/qingdao.jpg");
	cvtColor(src_img, src_img, COLOR_RGB2GRAY);
	Mat enhance_filter_img = Mat::zeros(src_img.size(), src_img.type());
	get_enhance_filter_img(src_img, enhance_filter_img, Size(5, 5), 0.5);
	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", src_img);
	namedWindow("高提升滤波", WINDOW_NORMAL);
	imshow("高提升滤波", enhance_filter_img);
}

// Laplacian、Robert、Sobel模板锐化图像
void demo8()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat laplacian_img = Mat::zeros(src_img.size(), src_img.type());
	Mat rob_img = Mat::zeros(src_img.size(), src_img.type());
	Mat sob_img = Mat::zeros(src_img.size(), src_img.type());

	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", src_img);

	//Laplacian
	get_laplacian_img(src_img, laplacian_img);
	namedWindow("Laplacian", WINDOW_NORMAL);
	imshow("Laplacian", laplacian_img);

	//Robert
	int x[3][3] = { 0, 0, 0, 0, -1, 0, 0, 0, 1 };
	int y[3][3] = { 0, 0, 0, 0, 0, -1, 0, 1, 0 };
	get_rob_sob_img(src_img, rob_img, x, y);
	namedWindow("Robert", WINDOW_NORMAL);
	imshow("Robert", rob_img);

	//Sobel
	int _x[3][3] = { -1, -2, -1, 0, 0, 0, -1, -2, -1 };
	int _y[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	get_rob_sob_img(src_img, sob_img, x, y);
	namedWindow("Sobel", WINDOW_NORMAL);
	imshow("Sobel", sob_img);
}

// 图像噪声
void demo9()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat salt_img = salt_pepper_noise(src_img, 10000, 255);
	Mat pepper_img = salt_pepper_noise(src_img, 10000, 0);
	Mat salt_pepper_img = salt_pepper_noise(salt_img, 10000, 0);
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("盐噪声", WINDOW_NORMAL);
	imshow("盐噪声", salt_img);
	namedWindow("椒噪声", WINDOW_NORMAL);
	imshow("椒噪声", pepper_img);
	namedWindow("椒盐噪声", WINDOW_NORMAL);
	imshow("椒盐噪声", salt_pepper_img);
	namedWindow("高斯噪声", WINDOW_NORMAL);
	imshow("高斯噪声", gauss_noise_img);
}

void demo10()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat salt_img = salt_pepper_noise(src_img, 10000, 255);
	Mat pepper_img = salt_pepper_noise(src_img, 10000, 0);
	Mat salt_pepper_img = salt_pepper_noise(salt_img, 10000, 0);
	namedWindow("椒盐噪声", WINDOW_NORMAL);
	imshow("椒盐噪声", salt_pepper_img);
	Mat arithmetic_mean_img = filter_process(salt_pepper_img, arithmetic_mean_kernel, 5);
	namedWindow("算术平均滤波", WINDOW_NORMAL);
	imshow("算术平均滤波", arithmetic_mean_img);
	Mat geometric_mean_img = filter_process(salt_pepper_img, geometric_mean_kernel, 5);
	namedWindow("几何平均滤波", WINDOW_NORMAL);
	imshow("几何平均滤波", geometric_mean_img);
	Mat harmonic_mean_img = filter_process(salt_pepper_img, harmonic_mean_kernel, 5);
	namedWindow("谐波平均滤波", WINDOW_NORMAL);
	imshow("谐波平均滤波", harmonic_mean_img);
	Mat antiharmonic_mean_img = filter_process(salt_pepper_img, antiharmonic_mean_kernel, 5);
	namedWindow("反谐波平均滤波", WINDOW_NORMAL);
	imshow("反谐波平均滤波", antiharmonic_mean_img);
}

void demo11()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("高斯噪声", WINDOW_NORMAL);
	imshow("高斯噪声", gauss_noise_img);
	Mat arithmetic_mean_img = filter_process(gauss_noise_img, arithmetic_mean_kernel, 5);
	namedWindow("算术平均滤波", WINDOW_NORMAL);
	imshow("算术平均滤波", arithmetic_mean_img);
	Mat geometric_mean_img = filter_process(gauss_noise_img, geometric_mean_kernel, 5);
	namedWindow("几何平均滤波", WINDOW_NORMAL);
	imshow("几何平均滤波", geometric_mean_img);
	Mat harmonic_mean_img = filter_process(gauss_noise_img, harmonic_mean_kernel, 5);
	namedWindow("谐波平均滤波", WINDOW_NORMAL);
	imshow("谐波平均滤波", geometric_mean_img);
	Mat antiharmonic_mean_img = filter_process(gauss_noise_img, antiharmonic_mean_kernel, 5);
	namedWindow("反谐波平均滤波", WINDOW_NORMAL);
	imshow("反谐波平均滤波", geometric_mean_img);
}

void demo12()
{
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, Size(300, 300));
	Mat salt_img = salt_pepper_noise(src_img, 4000, 255);
	Mat salt_pepper_img = salt_pepper_noise(salt_img, 4000, 0);
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("椒盐噪声", WINDOW_NORMAL);
	imshow("椒盐噪声", salt_pepper_img);
	namedWindow("高斯噪声", WINDOW_NORMAL);
	imshow("高斯噪声", gauss_noise_img);
	Mat salt_pepper_median_img = filter_process(salt_pepper_img, median_kernel, 3);
	namedWindow("椒盐噪声中值滤波", WINDOW_NORMAL);
	imshow("椒盐噪声中值滤波", salt_pepper_median_img);
	Mat gauss_median_img = filter_process(gauss_noise_img, median_kernel, 3);
	namedWindow("高斯噪声中值滤波", WINDOW_NORMAL);
	imshow("高斯噪声中值滤波", gauss_median_img);
}

void demo13()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("高斯噪声", WINDOW_NORMAL);
	imshow("高斯噪声", gauss_noise_img);
	Mat gauss_adaptive_mean_img = filter_process(gauss_noise_img, adaptive_mean_kernel, 5);
	namedWindow("高斯噪声自适应均值滤波", WINDOW_NORMAL);
	imshow("高斯噪声自适应均值滤波", gauss_adaptive_mean_img);
}

void demo14()
{
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, Size(300, 300));
	Mat salt_img = salt_pepper_noise(src_img, 5000, 255);
	Mat salt_pepper_img = salt_pepper_noise(salt_img, 5000, 0);
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("椒盐噪声", WINDOW_NORMAL);
	imshow("椒盐噪声", salt_pepper_img);
	namedWindow("高斯噪声", WINDOW_NORMAL);
	imshow("高斯噪声", gauss_noise_img);
	Mat salt_pepper_adaptive_median_img = filter_process(salt_pepper_img, adaptive_median_kernel, 3);
	namedWindow("椒盐噪声自适应中值滤波", WINDOW_NORMAL);
	imshow("椒盐噪声自适应中值滤波", salt_pepper_adaptive_median_img);
	Mat gauss_adaptive_median_img = filter_process(gauss_noise_img, adaptive_median_kernel, 3);
	namedWindow("高斯噪声自适应中值滤波", WINDOW_NORMAL);
	imshow("高斯噪声自适应中值滤波", gauss_adaptive_median_img);
}

int main()
{
	cout << "1:二值变换" << endl << "2:灰度变换" << endl << "3:补色变换" << endl << "4:彩色图和灰度图的直方图均衡化" << endl << "5:灰度图的直方图规定化" << endl << "6:均值平滑处理和高斯平滑处理" << endl
		<< "7:高提升滤波算法" << endl << "8:Laplacian、Robert、Sobel模板锐化图像" << endl << "9:图像噪声" << endl << "10:椒盐噪声图的均值滤波" << endl << "11:高斯噪声的均值滤波" << endl
		<< "12:中值滤波" << endl << "13:自适应均值滤波" << endl << "14:自适应中值滤波" << endl;
	int flag;
	cin >> flag;
	switch (flag)
	{
	case 1:
		demo1();
		break;
	case 2:
		demo2();
		break;
	case 3:
		demo3();
		break;
	case 4:
		demo4();
		break;
	case 5:
		demo5();
		break;
	case 6:
		demo6();
		break;
	case 7:
		demo7();
		break;
	case 8:
		demo8();
		break;
	case 9:
		demo9();
		break;
	case 10:
		demo10();
		break;
	case 11:
		demo11();
		break;
	case 12:
		demo12();
		break;
	case 13:
		demo13();
		break;
	case 14:
		demo14();
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