#include "noise.h"


// value=0   �� 
// value=255 ��
Mat salt_pepper_noise(const Mat& input, int n, int value) {
    int x, y;
    int row = input.rows, col = input.cols;
    bool color = input.channels() > 1;
    Mat target = input.clone();
    for (int i = 0; i < n; i++) {
        x = rand() % row;
        y = rand() % col;
        if (color) {
            target.at<Vec3b>(x, y)[0] = value;
            target.at<Vec3b>(x, y)[1] = value;
            target.at<Vec3b>(x, y)[2] = value;
        }
        else {
            target.at<uchar>(x, y) = value;
        }
    }
    return target;
}

//��ȡ�����˹����ͼ��
Mat gauss_noise(const Mat& input, int avg, int sd) {
    Mat noise = Mat::zeros(input.size(), input.type());
    //Opencv�������
    RNG rng(rand());
    //��˹�ֲ���ƽ��������ȷֲ�����Сֵ
    int a = avg;
    //��˹�ֲ��ı�׼�����ȷֲ������ֵ
    int b = sd;
    //���ͣ�NORMALΪ��˹�ֲ���UNIFORMΪ���ȷֲ�
    int distType = RNG::NORMAL;
    //����һ�����ϸ�˹�ֲ�������ȷֲ�����ͼ��
    rng.fill(noise, RNG::NORMAL, a, b);
    Mat output = input.clone();
    output = output + noise;
    return output;
}

//��ͼ������˲�����
Mat filter_process(const Mat& img, uchar kernel(const Mat&, int, int, int, int), int size, int q) {
    Mat target = img.clone();
    vector<Mat> channels;
    int m = size / 2;
    split(target, channels);
    for (Mat ch : channels) {
        Mat src = ch.clone();
        for (int i = src.rows - size; i >= 0; i--)
            for (int j = src.cols - size; j >= 0; j--) {
                ch.at<uchar>(i + m, j + m) = kernel(src, size, i, j, q);
            }
    }
    merge(channels, target);
    return target;
}

//������ֵ����
uchar arithmetic_mean_kernel(const Mat& img, int size, int i, int j, int q) {
    double res = 0;
    for (int r = i + size - 1; r >= i; r--) {
        for (int c = j + size - 1; c >= j; c--) {
            res += img.at<uchar>(r, c);
        }
    }
    return (uchar)(res / ((double)size * size));
}

//���ξ�ֵ����
uchar geometric_mean_kernel(const Mat& img, int size, int i, int j, int q) {
    double res = 1;
    for (int r = i + size - 1; r >= i; r--) {
        for (int c = j + size - 1; c >= j; c--) {
            res *= img.at<uchar>(r, c);
        }
    }
    return (uchar)(pow(res, 1.0 / ((double)size * size)));
}

//г��ƽ��ֵ����
uchar harmonic_mean_kernel(const Mat& img, int size, int i, int j, int q) {
    double res = 0;
    for (int r = i + size - 1; r >= i; r--) {
        for (int c = j + size - 1; c >= j; c--) {
            res += 1.0 / ((double)img.at<uchar>(r, c) + 1);
        }
    }
    return (uchar)(((double)size * size) / res - 1);
}

//��г��ƽ��ֵ����
uchar antiharmonic_mean_kernel(const Mat& img, int size, int i, int j, int q) {
    double res1 = 0, res2 = 0;
    for (int r = i + size - 1; r >= i; r--) {
        for (int c = j + size - 1; c >= j; c--) {
            res1 += pow(img.at<uchar>(r, c), q + 1);
            res2 += pow(img.at<uchar>(r, c), q);
        }
    }
    return (uchar)(res1 / res2);
}

//����ѡ����ֵ
int quick_select(vector<uchar>& vec, int left, int right, int target) {
    int t = vec[left], l = left, r = right;
    bool isLeft = true;
    while (l < r) {
        if (isLeft) {
            while (l < r && vec[r] >= t)
                r--;
            vec[l] = vec[r];
        }
        else {
            while (l < r && vec[l] < t)
                l++;
            vec[r] = vec[l];
        }
        isLeft = !isLeft;
    }
    vec[l] = t;
    if (l < target)
        return quick_select(vec, l + 1, right, target);
    else if (l > target)
        return quick_select(vec, left, l - 1, target);
    return vec[l];
}

//��ֵ����
uchar median_kernel(const Mat& img, int size, int i, int j, int q) {
    vector<uchar> values;
    for (int r = i + size - 1; r >= i; r--) {
        for (int c = j + size - 1; c >= j; c--) {
            values.push_back(img.at<uchar>(r, c));
        }
    }
    int cnt = size * size;
    return quick_select(values, 0, cnt - 1, cnt / 2);
}

uchar adaptive_mean_kernel(const Mat& img, int size, int i, int j, int q) {
    double avg = 0, ds = 0;
    uchar u = img.at<uchar>(i + size / 2, j + size / 2);
    for (int r = i + size - 1; r >= i; r--) {
        for (int c = j + size - 1; c >= j; c--) {
            avg += img.at<uchar>(r, c);
        }
    }
    avg /= size * size;
    for (int r = i + size - 1; r >= i; r--) {
        for (int c = j + size - 1; c >= j; c--) {
            ds += pow(avg - img.at<uchar>(r, c), 2);
        }
    }
    double rate = q / ds;
    if (rate > 1.0)
        rate = 1.0;
    return (int)(u - rate * (u - avg));
}

uchar adaptive_median_kernel(const Mat& img, int size, int i, int j, int max_size) {
    vector<uchar> values;
    for (int r = i + size - 1; r >= i; r--) {
        for (int c = j + size - 1; c >= j; c--) {
            values.push_back(img.at<uchar>(r, c));
        }
    }
    int cnt = size * size;
    int mid = quick_select(values, 0, cnt - 1, cnt / 2);
    int min = quick_select(values, 0, cnt - 1, 0);
    int max = quick_select(values, 0, cnt - 1, cnt - 1);
    if (mid < max && min < mid) {
        uchar u = img.at<uchar>(i + size / 2, j + size / 2);
        return (min < u&& u < max) ? u : mid;
    }
    if (size == max_size || i == 0 || j == 0 || i + size + 1 >= img.rows || j + size + 1 >= img.cols)
        return mid;
    return adaptive_median_kernel(img, size + 2, i - 1, j - 1, max_size);
}