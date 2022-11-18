#include "opencv2/opencv.hpp"

int main()
{
	cv::Mat gray = cv::imread("sub.png", cv::IMREAD_GRAYSCALE);
	cv::Mat mask;
	// ��ֵ����5����������Ϊ��ԭͼ��Ŀ��ͼ������ͼ������ֵ�����ֵ����ֵ������
	// ���������ǰѴ�����ֵ������ֵ��Ϊ255������Ϊ0�������һ������ΪTHRESH_BINARY
	// �Ѵ�����ֵ������ֵ��Ϊ0������Ϊ255�������һ������ΪTHRESH_BINARY_INV
	cv::threshold(gray, mask, 250, 255, cv::THRESH_BINARY_INV);
	cv::imshow("mask", mask);
	cv::imwrite("mask.jpg", mask); // �˴�����������ͼ�����Կ�һ�����ͼƬ

	cv::Mat img = cv::imread("sub.png");
	cv::Mat frame;
	cv::VideoCapture cap(0);
	while (cap.isOpened())
	{
		cap >> frame;
		img.copyTo(frame(cv::Rect(100, 100, img.cols, img.rows)), mask);
		cv::imshow("video", frame);
		cv::waitKey(1);	
	}
	return 1;
}
