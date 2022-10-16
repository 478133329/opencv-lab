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
	Mat src_img = imread("qingdao.jpg");
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
	Mat src_img = imread("qingdao.jpg");
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
	Mat src_img = imread("qingdao.jpg");
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
	Mat src_img = imread("qingdao.jpg");
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

// ͼ��ƽ������
void demo6() {
	Mat src_img = imread("img/qingdao.jpg");
	Mat mean_smoothing_img = Mat::zeros(src_img.size(), src_img.type());
	get_mean_smoothing_img(src_img, mean_smoothing_img, Size(3, 3));
	namedWindow("ԭͼ", WINDOW_NORMAL);
	imshow("ԭͼ", src_img);
	namedWindow("��ֵƽ������ͼ", WINDOW_NORMAL);
	imshow("��ֵƽ������ͼ", mean_smoothing_img);
}

// ��ɫͼ��ƽ������
void demo7() {
	Mat src_img = imread("img/qingdao.jpg");
	Mat mask, gauss_smoothing_img;
	get_gauss_mask(mask, Size(5, 5), 0.8);
	get_gauss_smoothing_img(src_img, gauss_smoothing_img, mask);
	namedWindow("ԭͼ", WINDOW_NORMAL);
	imshow("ԭͼ", src_img);
	namedWindow("��˹ƽ������ͼ", WINDOW_NORMAL);
	imshow("��˹ƽ������ͼ", gauss_smoothing_img);
}

int main()
{
	cout << "1:��ֵ�任" << endl << "2:�Ҷȱ任" << endl << "3:��ɫ�任" << endl << "4:��ɫͼ�ͻҶ�ͼ��ֱ��ͼ���⻯" << endl << "5:�Ҷ�ͼ��ֱ��ͼ�涨��" << endl << "6:�Ҷ�ͼ��ƽ������" << endl
		<< "7:��ɫͼ��ƽ������" << endl;
	int flag;
	cin >> flag;
	switch (flag)
	{
	case 1:
		// �����޸���ֵ������ͼƬչʾ��һ����
		demo1();
		break;
	case 2:
		// ���������������������ʾ��
		demo2();
		break;
	case 3:
		// �ϳ�һ��ͼ
		demo3();
		break;
	case 4:
		// �ϳ�һ��ͼ����չʾ�Ҷ�ͼ�Ͳ�ɫͼ��ֱ��ͼ�ֲ�
		demo4();
		break;
	case 5:
		// �ϳ�һ��ͼ
		demo5();
		break;
	case 6:
		// �Ҷ�ͼ��ƽ������
		demo6();
		break;
	case 7:
		// ��ɫͼ��ƽ������
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