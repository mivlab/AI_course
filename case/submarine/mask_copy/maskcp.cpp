#include "opencv2/opencv.hpp"

int main()
{
	cv::Mat gray = cv::imread("sub.png", cv::IMREAD_GRAYSCALE);
	cv::Mat mask;
	// 二值化，5个参数依次为：原图、目标图（掩码图）、阈值、最大值、二值化类型
	// 它的作用是把大于阈值的像素值设为255，否则为0，当最后一个参数为THRESH_BINARY
	// 把大于阈值的像素值设为0，否则为255，当最后一个参数为THRESH_BINARY_INV
	cv::threshold(gray, mask, 250, 255, cv::THRESH_BINARY_INV);
	cv::imshow("mask", mask);
	cv::imwrite("mask.jpg", mask); // 此处保存了掩码图，可以看一下这个图片

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
