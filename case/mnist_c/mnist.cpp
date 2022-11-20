#include <stdio.h>
#include "opencv2/opencv.hpp"

void read_conv_param(const char *file, float weight[], float bias[])
{
	FILE* fp = fopen(file, "rt");
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 27; j++)
		{
			fscanf(fp, "%f ", &weight[27 * i + j]);
		}
		fscanf(fp, "\n");
		fscanf(fp, "%f\n", &bias[i]);
	}
	fclose(fp);
}


int main()
{
	float weight[16 * 27], bias[16];
	read_conv_param("1.txt", weight, bias);
	cv::Mat img = cv::imread("2.jpg");
	cv::resize(img, img, cv::Size(28, 28));
	cv::Mat conv_out(cv::Size(28, 28), CV_32FC1); // 创建一个28x28的32位浮点数矩阵，单通道
	for (int out_c = 0; out_c < 16; out_c++) // 输出16通道
	{
		// 我们没有做填充，所以输出有效宽高要少2行2列
		for (int i = 0; i < img.rows - 2; i++) // 图像高。
		{
			for (int j = 0; j < img.cols - 2; j++) // 图像宽
			{
				// 接下来做3x3卷积
				float out = 0;
				for (int in_c = 0; in_c < 3; in_c++) // 输入3通道
				{
					for (int m = 0 ; m < 3; m++) // 卷积核大小3x3
					{
						for (int n = 0; n < 3; n++) 
							out += img.at<cv::Vec3b>(i + m, j + n)[in_c] / 255.0f * weight[out_c * 27 + in_c * 9 + m * 3 + n];
					}
				}
				out += bias[out_c];
				conv_out.at<float>(i + 1, j + 1) = out; // i,j对应的是卷积核的左上角，它的中心是i+1, j+1
				// 为每个通道输出一次结果，此2行仅为显示计算结果，可去掉
				if (i == img.rows - 3 && j == img.cols - 3)
					printf("channel %d %f\n", out_c, out);
			}
		}
	}
}