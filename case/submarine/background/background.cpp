#include "opencv2/opencv.hpp"
int main() {
	cv::VideoCapture cap(0); // 摄像头
	cv::Mat img = cv::imread("11.png"); // 柱子图
	cv::Mat frame; // 摄像头画面
	int x = 0;// 从柱子图最左边开始截取

	while (true) {
		cap >> frame; // 采集一帧图像
		int win = 300; //截取窗口宽度
		cv::Mat img1 = img(cv::Rect(x, 0, 300, img.rows));
		cv::resize(img1, img1, frame.size()); // 缩放到相同大小
		uchar* pframe = frame.data; //摄像头图像数据起始地址指针
		uchar* pimg = img1.data;//柱子图像数据起始地址指针
		for(int i = 0; i < img1.rows; i++) // 循环图像每一行
			for (int j = 0; j < img1.cols; j++)// 循环图像每一行
			{
				// 判断柱子图是否为白色区域。仅对非白色区域，才把摄像头的像素覆盖掉
				if (pimg[i * img1.cols * 3 + j * 3 + 0] + 
					pimg[i * img1.cols * 3 + j * 3 + 1] +
					pimg[i * img1.cols * 3 + j * 3 + 2] < 200 * 3)
				{
					// BGR（蓝绿红）三个通道依次赋值
					pframe[i * img1.cols * 3 + j * 3 + 0] = pimg[i * img1.cols * 3 + j * 3 + 0];
					pframe[i * img1.cols * 3 + j * 3 + 1] = pimg[i * img1.cols * 3 + j * 3 + 1];
					pframe[i * img1.cols * 3 + j * 3 + 2] = pimg[i * img1.cols * 3 + j * 3 + 2];
				}
			}
		cv::imshow("image", frame); // 显示图像
		x += 3; // 柱子图每帧滑动步长
		if (x > img.cols - win) // 当到达图像右边界，转头
			x = 0;
		cv::waitKey(1); // 等待1毫秒，以便能观看视频
	}
	return 1;
}

