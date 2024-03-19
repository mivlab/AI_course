#include "opencv2/opencv.hpp"
#define PI 3.14159
int main()
{
	float theta = 15 / 180.0f * PI;
	cv::Mat K = (cv::Mat_<float>(3, 3) << 960, 0, 959.5, 0, 960, 539.5, 0, 0, 1);
	cv::Mat W_C_R = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, -sin(theta),
		cos(theta), 0, -cos(theta), -sin(theta));
	cv::Mat W_C_t = (cv::Mat_<float>(3, 1) << 0, -1.2, 1.5);
	cv::Mat m = (cv::Mat_<float>(3, 1) << 1065, 530, 1); // ǰ�����ײ�����λ����������

	// �����ڲξ�����棬�õ�һ������
	cv::Mat invK;
	cv::invert(K, invK);
	cv::Mat ray = invK * m;

	// �ѹ���ת��������ǰ���������������ϵ��
	cv::Mat R = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, cos(theta), sin(theta),
		0, -sin(theta), cos(theta));
	cv::Mat ray1 = R * ray;
	
	// ��y��������߶ȣ����ݱ�����ϵ���x��z
	float x = ray1.at<float>(0, 0) / ray1.at<float>(1, 0) * 1.5f;
	float z = ray1.at<float>(2, 0) / ray1.at<float>(1, 0) * 1.5f;
	float y = 1.5f;
	
	std::cout << x << "," << y << ", " << z << std::endl;
	// ת������������ϵ�£����ǰ��λ�ڳ�������ϵ��x = 0.66, y=5.8-1.2����
}