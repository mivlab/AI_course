#include "opencv2/opencv.hpp"

int main()
{
	cv::Mat gray = cv::imread("sub.png", cv::IMREAD_GRAYSCALE);
	cv::Mat mask;
	cv::threshold(gray, mask, 250, 255, cv::THRESH_BINARY_INV);
	cv::imshow("mask", mask);

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