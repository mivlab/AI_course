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
	cv::Mat conv_out(cv::Size(28, 28), CV_32FC1); // ����һ��28x28��32λ���������󣬵�ͨ��
	for (int out_c = 0; out_c < 16; out_c++) // ���16ͨ��
	{
		// ����û������䣬���������Ч���Ҫ��2��2��
		for (int i = 0; i < img.rows - 2; i++) // ͼ��ߡ�
		{
			for (int j = 0; j < img.cols - 2; j++) // ͼ���
			{
				// ��������3x3���
				float out = 0;
				for (int in_c = 0; in_c < 3; in_c++) // ����3ͨ��
				{
					for (int m = 0 ; m < 3; m++) // ����˴�С3x3
					{
						for (int n = 0; n < 3; n++) 
							out += img.at<cv::Vec3b>(i + m, j + n)[in_c] / 255.0f * weight[out_c * 27 + in_c * 9 + m * 3 + n];
					}
				}
				out += bias[out_c];
				conv_out.at<float>(i + 1, j + 1) = out; // i,j��Ӧ���Ǿ���˵����Ͻǣ�����������i+1, j+1
				// Ϊÿ��ͨ�����һ�ν������2�н�Ϊ��ʾ����������ȥ��
				if (i == img.rows - 3 && j == img.cols - 3)
					printf("channel %d %f\n", out_c, out);
			}
		}
	}
}