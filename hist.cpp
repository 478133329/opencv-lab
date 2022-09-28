#include "hist.h"

Mat get_hist(const Mat& mat)
{
	Mat hist;
	int channels[] = { 0 };
	int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 255 };
	const float* ranges[] = { range };
	calcHist(&mat, 1, channels, Mat(), hist, 1, hist_size, ranges, true, false);
	return hist;
}

Mat get_hist_img(Mat& mat, int bins)
{
	double max_val;
	Mat hist = get_hist(mat);
	minMaxLoc(hist, 0, &max_val, 0, 0);
	Mat hist_img(Size(bins * 2, bins * 2), CV_8UC1);
	for (int i = 0; i < bins; i++)
	{
		int height = cvRound(hist.at<float>(i) / max_val * bins * 2);
		rectangle(hist_img, Point(i * 2, bins * 2), Point((i + 1) * 2, bins * 2 - height), Scalar(0, 0, 0));
	}
	return hist_img;
}

Mat get_equalized_gray_img(Mat& src_img, int bins)
{
	vector<int> nk(bins, 0);
	unordered_map<int, int> m;
	Mat dst_img = src_img.clone();
	for (int i = 0; i < dst_img.total(); i++)
	{
		nk[dst_img.data[i]]++;
	}
	for (int i = 1; i < bins; i++)
	{
		nk[i] += nk[i - 1];
	}
	for (int i = 0; i < bins; i++)
	{
		int temp = cvRound(nk[i] / (double)dst_img.total() * 255);
		m[i] = temp;
	}
	for (int i = 0; i < dst_img.total(); i++)
	{
		dst_img.data[i] = m[dst_img.data[i]];
		// src_img.at<int>(i) = m[src_img.at<int>(i)]; 报错，int占32位，原图是8位。
	}
	return dst_img;
}

Mat get_equalized_rgb_img(Mat& src_img, int bins)
{
	int channels = src_img.channels();
	Mat dst_img = src_img.clone();
	for (int ch = 0; ch < channels; ch++)
	{
		vector<int> nk(bins, 0);
		unordered_map<int, int> m;
		for (int i = 0; i < dst_img.rows; i++)
		{
			for (int j = 0; j < dst_img.cols; j++)
			{
				nk[dst_img.at<Vec3b>(i, j)[ch]]++;
			}
		}
		for (int i = 1; i < bins; i++)
		{
			nk[i] += nk[i - 1];
		}
		for (int i = 0; i < bins; i++)
		{
			int temp = cvRound(nk[i] / (double)dst_img.total() * 255);
			m[i] = temp;
		}
		for (int i = 0; i < dst_img.rows; i++)
		{
			for (int j = 0; j < dst_img.cols; j++)
			{
				dst_img.at<Vec3b>(i, j)[ch] = m[dst_img.at<Vec3b>(i, j)[ch]];
			}
		}
	}
	return dst_img;

}
unordered_map<int, int> get_equal_map(Mat hist)
{
	vector<int> nk(hist.total(), 0);
	unordered_map<int, int> m;
	int total = 0;
	for (int i = 0; i < hist.total(); i++)
	{
		total += hist.at<float>(i);
	}
	for (int i = 0; i < hist.total(); i++)
	{
		nk[i] = hist.at<float>(i);
	}
	for (int i = 1; i < hist.total(); i++)
	{
		nk[i] += nk[i - 1];
	}
	for (int i = 0; i < hist.total(); i++)
	{
		int temp = cvRound(nk[i] / (double)total * 255);
		m[i] = temp;
	}
	return m;
}

unordered_map<int, int> reverse_map(unordered_map<int, int> m)
{
	unordered_map<int, int> rm;
	for (auto iter = m.begin(); iter != m.end(); ++iter)
	{
		rm[iter->second] = max(rm[iter->second], iter->first);
	}
	return rm;
}

Mat get_matched_img(Mat& src_img, const Mat& target_img)
{
	Mat hist1 = get_hist(src_img);
	Mat hist2 = get_hist(target_img);
	Mat dst_img = src_img.clone();
	unordered_map<int, int> m1 = get_equal_map(hist1);
	unordered_map<int, int> m2 = get_equal_map(hist2);
	auto m = reverse_map(m2);
	int last = 0;
	for (int i = 0; i < dst_img.total(); i++)
	{
		int temp = m1[dst_img.data[i]];
		if (m.find(temp) != m.end())
		{
			dst_img.data[i] = m[temp];
			last = m[temp];
		}
		else
		{
			dst_img.data[i] = last;
		}
	}
	return dst_img;
}