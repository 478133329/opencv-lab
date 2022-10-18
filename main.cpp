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
	namedWindow("�Ա�", WINDOW_NORMAL);
	imshow("�Ա�", dst);
}

// ��ֵ�任
void demo1()
{
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, Size(1280, 960));
	cvtColor(src_img, src_img, COLOR_RGB2GRAY);
	Mat bin_img = bin_val(src_img, 128);
	namedWindow("ԭͼ", WINDOW_NORMAL);
	imshow("ԭͼ", src_img);
	namedWindow("�ڰ�ͼ", WINDOW_NORMAL);
	imshow("�ڰ�ͼ", bin_img);
}

// �Ҷȱ任�������任��٤��任��
void demo2()
{
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, Size(1280, 960));
	cvtColor(src_img, src_img, COLOR_RGB2GRAY);
	Mat log_img = log_trans(src_img, 10);
	Mat gamma_img = gamma_trans(src_img, 1.5, 0.8);
	vector<Mat*> images;                // vector���Դ��ָ�룬���ܴ�����á�
	images.push_back(&src_img);
	images.push_back(&log_img);
	images.push_back(&gamma_img);
	showImgsAndHistImgs(images);
}

// ��ɫ�任
void demo3()
{
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, Size(1280, 960));
	Mat complement_img = complement_trans(src_img);
	namedWindow("ԭͼ", WINDOW_NORMAL);
	imshow("ԭͼ", src_img);
	namedWindow("��ɫͼ��", WINDOW_NORMAL);
	imshow("��ɫͼ��", complement_img);
}

// ԭͼ�Ĳ�ɫ����ͼ�ͻҶȾ���ͼ
void demo4()
{
	int bins = 256;
	Size size(300, 300);
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, size);
	namedWindow("ԭͼ", WINDOW_AUTOSIZE);
	imshow("ԭͼ", src_img);

	Mat dst = get_equalized_rgb_img(src_img, 256);
	namedWindow("ԭͼ����ͼ", WINDOW_AUTOSIZE);
	imshow("ԭͼ����ͼ", dst);

	cvtColor(src_img, src_img, COLOR_RGB2GRAY);
	namedWindow("ԭͼ�Ҷ�ͼ", WINDOW_AUTOSIZE);
	imshow("ԭͼ�Ҷ�ͼ", src_img);

	Mat equal_src_img = get_equalized_gray_img(src_img, bins);
	Mat equal_src_img_hist_img = get_hist_img(equal_src_img, bins);
	namedWindow("ԭͼ�Ҷ�ͼ���⻯", WINDOW_AUTOSIZE);
	imshow("ԭͼ�Ҷ�ͼ���⻯", equal_src_img);

}

// ԭͼ�涨��
void demo5()
{
	int bins = 256;
	Size size(1280, 960);
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, size);
	cvtColor(src_img, src_img, COLOR_RGB2GRAY);

	namedWindow("ԭͼ�Ҷ�ͼ", WINDOW_NORMAL);
	imshow("ԭͼ�Ҷ�ͼ", src_img);

	Mat src_img_hist_img = get_hist_img(src_img, bins);
	namedWindow("ԭͼ�Ҷ�ͼֱ��ͼ", WINDOW_NORMAL);
	imshow("ԭͼ�Ҷ�ͼֱ��ͼ", src_img_hist_img);

	Mat target_img = imread("background.jpg");
	resize(target_img, target_img, size);
	cvtColor(target_img, target_img, COLOR_RGB2GRAY);
	Mat target_img_hist_img = get_hist_img(target_img, bins);
	namedWindow("Ŀ��ͼֱ��ͼ", WINDOW_NORMAL);
	imshow("Ŀ��ͼֱ��ͼ", target_img_hist_img);

	Mat matched_img = get_matched_img(src_img, target_img);
	namedWindow("�涨��ԭͼ�Ҷ�ͼ", WINDOW_NORMAL);
	imshow("�涨��ԭͼ�Ҷ�ͼ", matched_img);

	Mat matched_img_hist_img = get_hist_img(matched_img, bins);
	namedWindow("�涨��ԭͼ�Ҷ�ͼֱ��ͼ", WINDOW_NORMAL);
	imshow("�涨��ԭͼ�Ҷ�ͼֱ��ͼ", matched_img_hist_img);
}

// ƽ������
void demo6()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat mean_smoothing_img = Mat::zeros(src_img.size(), src_img.type());
	get_mean_smoothing_img(src_img, mean_smoothing_img, Size(5, 5));
	Mat mask, gauss_smoothing_img;
	get_gauss_mask(mask, Size(5, 5), 0.8);
	get_gauss_smoothing_img(src_img, gauss_smoothing_img, mask);
	namedWindow("ԭͼ", WINDOW_NORMAL);
	imshow("ԭͼ", src_img);
	namedWindow("��ֵƽ������ͼ", WINDOW_NORMAL);
	imshow("��ֵƽ������ͼ", mean_smoothing_img);
	namedWindow("��˹ƽ������ͼ", WINDOW_NORMAL);
	imshow("��˹ƽ������ͼ", gauss_smoothing_img);
}

// �������˲��㷨
void demo7()
{
	Mat src_img = imread("img/qingdao.jpg");
	cvtColor(src_img, src_img, COLOR_RGB2GRAY);
	Mat enhance_filter_img = Mat::zeros(src_img.size(), src_img.type());
	get_enhance_filter_img(src_img, enhance_filter_img, Size(5, 5), 0.5);
	namedWindow("ԭͼ", WINDOW_NORMAL);
	imshow("ԭͼ", src_img);
	namedWindow("�������˲�", WINDOW_NORMAL);
	imshow("�������˲�", enhance_filter_img);
}

// Laplacian��Robert��Sobelģ����ͼ��
void demo8()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat laplacian_img = Mat::zeros(src_img.size(), src_img.type());
	Mat rob_img = Mat::zeros(src_img.size(), src_img.type());
	Mat sob_img = Mat::zeros(src_img.size(), src_img.type());

	namedWindow("ԭͼ", WINDOW_NORMAL);
	imshow("ԭͼ", src_img);

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

// ͼ������
void demo9()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat salt_img = salt_pepper_noise(src_img, 10000, 255);
	Mat pepper_img = salt_pepper_noise(src_img, 10000, 0);
	Mat salt_pepper_img = salt_pepper_noise(salt_img, 10000, 0);
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("������", WINDOW_NORMAL);
	imshow("������", salt_img);
	namedWindow("������", WINDOW_NORMAL);
	imshow("������", pepper_img);
	namedWindow("��������", WINDOW_NORMAL);
	imshow("��������", salt_pepper_img);
	namedWindow("��˹����", WINDOW_NORMAL);
	imshow("��˹����", gauss_noise_img);
}

void demo10()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat salt_img = salt_pepper_noise(src_img, 10000, 255);
	Mat pepper_img = salt_pepper_noise(src_img, 10000, 0);
	Mat salt_pepper_img = salt_pepper_noise(salt_img, 10000, 0);
	namedWindow("��������", WINDOW_NORMAL);
	imshow("��������", salt_pepper_img);
	Mat arithmetic_mean_img = filter_process(salt_pepper_img, arithmetic_mean_kernel, 5);
	namedWindow("����ƽ���˲�", WINDOW_NORMAL);
	imshow("����ƽ���˲�", arithmetic_mean_img);
	Mat geometric_mean_img = filter_process(salt_pepper_img, geometric_mean_kernel, 5);
	namedWindow("����ƽ���˲�", WINDOW_NORMAL);
	imshow("����ƽ���˲�", geometric_mean_img);
	Mat harmonic_mean_img = filter_process(salt_pepper_img, harmonic_mean_kernel, 5);
	namedWindow("г��ƽ���˲�", WINDOW_NORMAL);
	imshow("г��ƽ���˲�", harmonic_mean_img);
	Mat antiharmonic_mean_img = filter_process(salt_pepper_img, antiharmonic_mean_kernel, 5);
	namedWindow("��г��ƽ���˲�", WINDOW_NORMAL);
	imshow("��г��ƽ���˲�", antiharmonic_mean_img);
}

void demo11()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("��˹����", WINDOW_NORMAL);
	imshow("��˹����", gauss_noise_img);
	Mat arithmetic_mean_img = filter_process(gauss_noise_img, arithmetic_mean_kernel, 5);
	namedWindow("����ƽ���˲�", WINDOW_NORMAL);
	imshow("����ƽ���˲�", arithmetic_mean_img);
	Mat geometric_mean_img = filter_process(gauss_noise_img, geometric_mean_kernel, 5);
	namedWindow("����ƽ���˲�", WINDOW_NORMAL);
	imshow("����ƽ���˲�", geometric_mean_img);
	Mat harmonic_mean_img = filter_process(gauss_noise_img, harmonic_mean_kernel, 5);
	namedWindow("г��ƽ���˲�", WINDOW_NORMAL);
	imshow("г��ƽ���˲�", geometric_mean_img);
	Mat antiharmonic_mean_img = filter_process(gauss_noise_img, antiharmonic_mean_kernel, 5);
	namedWindow("��г��ƽ���˲�", WINDOW_NORMAL);
	imshow("��г��ƽ���˲�", geometric_mean_img);
}

void demo12()
{
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, Size(300, 300));
	Mat salt_img = salt_pepper_noise(src_img, 4000, 255);
	Mat salt_pepper_img = salt_pepper_noise(salt_img, 4000, 0);
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("��������", WINDOW_NORMAL);
	imshow("��������", salt_pepper_img);
	namedWindow("��˹����", WINDOW_NORMAL);
	imshow("��˹����", gauss_noise_img);
	Mat salt_pepper_median_img = filter_process(salt_pepper_img, median_kernel, 3);
	namedWindow("����������ֵ�˲�", WINDOW_NORMAL);
	imshow("����������ֵ�˲�", salt_pepper_median_img);
	Mat gauss_median_img = filter_process(gauss_noise_img, median_kernel, 3);
	namedWindow("��˹������ֵ�˲�", WINDOW_NORMAL);
	imshow("��˹������ֵ�˲�", gauss_median_img);
}

void demo13()
{
	Mat src_img = imread("img/qingdao.jpg");
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("��˹����", WINDOW_NORMAL);
	imshow("��˹����", gauss_noise_img);
	Mat gauss_adaptive_mean_img = filter_process(gauss_noise_img, adaptive_mean_kernel, 5);
	namedWindow("��˹��������Ӧ��ֵ�˲�", WINDOW_NORMAL);
	imshow("��˹��������Ӧ��ֵ�˲�", gauss_adaptive_mean_img);
}

void demo14()
{
	Mat src_img = imread("img/qingdao.jpg");
	resize(src_img, src_img, Size(300, 300));
	Mat salt_img = salt_pepper_noise(src_img, 5000, 255);
	Mat salt_pepper_img = salt_pepper_noise(salt_img, 5000, 0);
	Mat gauss_noise_img = gauss_noise(src_img, 10, 20);
	namedWindow("��������", WINDOW_NORMAL);
	imshow("��������", salt_pepper_img);
	namedWindow("��˹����", WINDOW_NORMAL);
	imshow("��˹����", gauss_noise_img);
	Mat salt_pepper_adaptive_median_img = filter_process(salt_pepper_img, adaptive_median_kernel, 3);
	namedWindow("������������Ӧ��ֵ�˲�", WINDOW_NORMAL);
	imshow("������������Ӧ��ֵ�˲�", salt_pepper_adaptive_median_img);
	Mat gauss_adaptive_median_img = filter_process(gauss_noise_img, adaptive_median_kernel, 3);
	namedWindow("��˹��������Ӧ��ֵ�˲�", WINDOW_NORMAL);
	imshow("��˹��������Ӧ��ֵ�˲�", gauss_adaptive_median_img);
}

int main()
{
	cout << "1:��ֵ�任" << endl << "2:�Ҷȱ任" << endl << "3:��ɫ�任" << endl << "4:��ɫͼ�ͻҶ�ͼ��ֱ��ͼ���⻯" << endl << "5:�Ҷ�ͼ��ֱ��ͼ�涨��" << endl << "6:��ֵƽ������͸�˹ƽ������" << endl
		<< "7:�������˲��㷨" << endl << "8:Laplacian��Robert��Sobelģ����ͼ��" << endl << "9:ͼ������" << endl << "10:��������ͼ�ľ�ֵ�˲�" << endl << "11:��˹�����ľ�ֵ�˲�" << endl
		<< "12:��ֵ�˲�" << endl << "13:����Ӧ��ֵ�˲�" << endl << "14:����Ӧ��ֵ�˲�" << endl;
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