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

void get_enhance_filter_img(const Mat& src, Mat& dst, Size ksize, double k) {
	Mat src_mean = src.clone();
	get_mean_smoothing_img(src, src_mean, ksize);
	int mask = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			mask = src.at<uchar>(i, j) - src_mean.at<uchar>(i, j);
			dst.at<uchar>(i, j) = src.at<uchar>(i, j) + mask * k;
		}
	}
}

void filter(const Mat& src, Mat& dst, int ksize, double** templateMatrix) {
	assert(src.channels() || src.channels() == 3);
	//使用模板进行平滑处理
	int border = ksize / 2;
	copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
	int channels = src.channels();
	int cols = dst.cols - border;
	int rows = dst.rows - border;
	for (int i = border; i < rows; i++) {
		for (int j = border; j < cols; j++) {
			double sum[3] = { 0 };
			for (int k = -border; k <= border; k++) {
				for (int m = -border; m <= border; m++) {
					if (channels == 1) {
						sum[0] += (double)templateMatrix[k + border][k + border] * dst.at<uchar>(i + k, j + m);
					}
					else if (channels == 3) {
						Vec3b rgb = dst.at<Vec3b>(i + k, j + m);
						auto tmp = templateMatrix[border + k][border + m];
						sum[0] += tmp * rgb[0];
						sum[1] += tmp * rgb[1];
						sum[2] += tmp * rgb[2];
					}
				}
			}
			//限定像素值在0-255之间
			for (int i = 0; i < channels; i++) {
				if (sum[i] < 0)
					sum[i] = 0;
				else if (sum[i] > 255)
					sum[i] = 255;
			}
			//
			if (channels == 1) {
				dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
			}
			else if (channels == 3) {
				//Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				Vec3b rgb;
				rgb[0] = static_cast<uchar>(sum[0]);
				rgb[1] = static_cast<uchar>(sum[1]);
				rgb[2] = static_cast<uchar>(sum[2]);

				dst.at<Vec3b>(i, j) = rgb;
			}
		}
	}
	for (int i = 0; i < ksize; i++)
		delete[] templateMatrix[i];
	delete[] templateMatrix;
}

void get_laplacian_img(const Mat& src, Mat& dst) {
	vector<double> list = { -1, -1, -1, -1, 4, -1, -1, -1, -1 };
	double** templateMatrix = new double* [3];
	for (int i = 0; i < 3; i++) {
		templateMatrix[i] = new double[3];
	}
	int k = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			templateMatrix[i][j] = list[k++];
		}
	}
	filter(src, dst, 3, templateMatrix);
}

void get_rob_sob_img(const Mat& src, Mat& dst, int x[][3], int y[][3]) {
	int border = 1;
	copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
	int channels = src.channels();
	int cols = dst.cols - border;
	int rows = dst.rows - border;
	for (int i = border; i < rows; i++) {
		for (int j = border; j < cols; j++) {
			int sum[3][3] = { 0 };
			for (int k = -border; k <= border; k++) {
				for (int m = -border; m <= border; m++) {
					if (channels == 1) {
						sum[0][0] += (int)x[k + border][k + border] * dst.at<uchar>(i + k, j + m);
						sum[0][1] += (int)y[k + border][k + border] * dst.at<uchar>(i + k, j + m);
					}
					else if (channels == 3) {
						Vec3b rgb = dst.at<Vec3b>(i + k, j + m);
						auto tmp1 = x[border + k][border + m];
						auto tmp2 = y[border + k][border + m];
						sum[0][0] += tmp1 * rgb[0];
						sum[1][0] += tmp1 * rgb[1];
						sum[2][0] += tmp1 * rgb[2];
						sum[0][1] += tmp2 * rgb[0];
						sum[1][1] += tmp2 * rgb[1];
						sum[2][1] += tmp2 * rgb[2];
					}
				}
			}
			//限定像素值在0-255之间
			for (int i = 0; i < channels; i++) {
				sum[i][2] = sum[i][0] + sum[i][1];
				if (sum[i][2] < 0)
					sum[i][2] = 0;
				else if (sum[i][3] > 255)
					sum[i][2] = 255;
			}
			//
			if (channels == 1) {
				dst.at<uchar>(i, j) = static_cast<uchar>(sum[0][2]);
			}
			else if (channels == 3) {
				//Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				Vec3b rgb;
				rgb[0] = static_cast<uchar>(sum[0][2]);
				rgb[1] = static_cast<uchar>(sum[1][2]);
				rgb[2] = static_cast<uchar>(sum[2][2]);

				dst.at<Vec3b>(i, j) = rgb;
			}
		}
	}
}

