#include "filtering.h"

void get_mean_smoothing_img(const Mat& src_img, Mat& dest_img, Size ksize)
{
	if (ksize.width % 2 == 0 || ksize.height % 2 == 0)
	{
		cout << "please input odd ksize!" << endl;
		exit(-1);
	}

	int awidth = (ksize.width - 1) / 2;
	int aheight = (ksize.height - 1) / 2;
	Mat asrc;
	copyMakeBorder(src_img, asrc, aheight, aheight, awidth, awidth, BORDER_DEFAULT);

	if (src_img.channels() == 1)
	{
		for (int i = aheight; i < src_img.rows + aheight; i++)
		{
			for (int j = awidth; j < src_img.cols + awidth; j++)
			{
				int sum = 0;
				int mean = 0;
				for (int k = i - aheight; k <= i + aheight; k++)
				{
					for (int l = j - awidth; l <= j + awidth; l++)
					{
						sum += asrc.at<uchar>(k, l);
					}
				}
				mean = sum / (ksize.width * ksize.height);
				dest_img.at<uchar>(i - aheight, j - awidth) = mean;
			}
		}
	}
	if (src_img.channels() == 3)
	{
		for (int i = aheight; i < src_img.rows + aheight; i++)
		{
			for (int j = awidth; j < src_img.cols + awidth; j++)
			{
				int sum[3] = { 0 };
				int mean[3] = { 0 };
				for (int k = i - aheight; k <= i + aheight; k++)
				{
					for (int l = j - awidth; l <= j + awidth; l++)
					{
						sum[0] += asrc.at<Vec3b>(k, l)[0];
						sum[1] += asrc.at<Vec3b>(k, l)[1];
						sum[2] += asrc.at<Vec3b>(k, l)[2];
					}
				}
				for (int m = 0; m < 3; m++)
				{
					mean[m] = sum[m] / (ksize.width * ksize.height);
					dest_img.at<Vec3b>(i - aheight, j - awidth)[m] = mean[m];
				}
			}
		}
	}
}

void get_gauss_mask(Mat& mask, Size ksize, double sigma)
{
	if (ksize.width % 2 == 0 || ksize.height % 2 == 0)
	{
		cout << "please input odd ksize!" << endl;
		exit(-1);
	}
	mask.create(ksize, CV_64F);
	int h = ksize.height;
	int w = ksize.width;
	int center_h = (ksize.height - 1) / 2;
	int center_w = (ksize.width - 1) / 2;
	double sum = 0;
	double x, y;
	for (int i = 0; i < h; i++)
	{
		x = pow(i - center_h, 2);
		for (int j = 0; j < w; j++)
		{
			y = pow(j - center_w, 2);
			mask.at<double>(i, j) = exp(-(x + y) / (2 * sigma * sigma));
			sum += mask.at<double>(i, j);
		}
	}
	mask = mask / sum;
}

void get_gauss_smoothing_img(const Mat& src, Mat& dst, Mat mask)
{
	int hh = (mask.rows - 1) / 2;
	int hw = (mask.cols - 1) / 2;
	dst = Mat::zeros(src.size(), src.type());

	Mat newsrc;
	copyMakeBorder(src, newsrc, hh, hh, hw, hw, BORDER_DEFAULT);

	for (int i = hh; i < src.rows + hh; i++)
	{
		for (int j = hw; j < src.cols + hw; j++)
		{
			double sum[3] = { 0 };
			for (int r = -hh; r <= hh; r++)
			{
				for (int c = -hw; c <= hw; c++)
				{
					if (src.channels() == 1)
					{
						sum[0] += newsrc.at<uchar>(i + r, j + c) * mask.at<double>(r + hh, c + hw);
					}
					else if (src.channels() == 3)
					{
						sum[0] += newsrc.at<Vec3b>(i + r, j + c)[0] * mask.at<double>(r + hh, c + hw);
						sum[1] += newsrc.at<Vec3b>(i + r, j + c)[1] * mask.at<double>(r + hh, c + hw);
						sum[2] += newsrc.at<Vec3b>(i + r, j + c)[2] * mask.at<double>(r + hh, c + hw);
					}
				}
			}
			for (int k = 0; k < src.channels(); k++)
			{
				if (sum[k] < 0)sum[k] = 0;
				else if (sum[k] > 255)sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(i - hh, j - hw) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3)
			{
				Vec3b rgb = { static_cast<uchar>(sum[0]) ,static_cast<uchar>(sum[1]) ,static_cast<uchar>(sum[2]) };
				dst.at<Vec3b>(i - hh, j - hw) = rgb;
			}
		}
	}
}
