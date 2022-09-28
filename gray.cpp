#include "gray.h"

// threshold(Mat& src, double th, double val1, double val2, Mat& dst);
Mat bin_val(Mat& src, int th)
{
	int total = src.total();
	Mat dst = src.clone();
	for (int i = 0; i < total; i++)
	{
		if (dst.data[i] > th)
		{
			dst.data[i] = 255;
		}
		else
		{
			dst.data[i] = 0;
		}
	}
	return dst;
}

Mat log_trans(Mat& src, int c)
{
	int total = src.total();
	Mat dst = src.clone();
	for (int i = 0; i < total; i++)
	{
		dst.data[i] = c * log(src.data[i] + 1);
	}
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	return dst;
}

Mat gamma_trans(Mat& src, int c, double gamma)
{
	int total = src.total();
	Mat dst = src.clone();
	for (int i = 0; i < total; i++)
	{
		dst.data[i] = c * pow(src.data[i], gamma);
	}
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	return dst;
}

Mat complement_trans(Mat& src)
{
	int total = src.total();
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int b = src.at<Vec3b>(i, j)[0];
			int g = src.at<Vec3b>(i, j)[1];
			int r = src.at<Vec3b>(i, j)[2];
			int maxrgb = max(max(r, g), b);
			int minrgb = min(min(r, g), b);
			dst.at<Vec3b>(i, j)[0] = maxrgb + minrgb - b;
			dst.at<Vec3b>(i, j)[1] = maxrgb + minrgb - g;
			dst.at<Vec3b>(i, j)[2] = maxrgb + minrgb - r;
		}
	}
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	return dst;
}