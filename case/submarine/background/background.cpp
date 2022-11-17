#include "opencv2/opencv.hpp"
int main() {
	cv::VideoCapture cap(0); // ����ͷ
	cv::Mat img = cv::imread("11.png"); // ����ͼ
	cv::Mat frame; // ����ͷ����
	int x = 0;// ������ͼ����߿�ʼ��ȡ

	while (true) {
		cap >> frame; // �ɼ�һ֡ͼ��
		int win = 300; //��ȡ���ڿ��
		cv::Mat img1 = img(cv::Rect(x, 0, 300, img.rows));
		cv::resize(img1, img1, frame.size()); // ���ŵ���ͬ��С
		uchar* pframe = frame.data; //����ͷͼ��������ʼ��ַָ��
		uchar* pimg = img1.data;//����ͼ��������ʼ��ַָ��
		for(int i = 0; i < img1.rows; i++) // ѭ��ͼ��ÿһ��
			for (int j = 0; j < img1.cols; j++)// ѭ��ͼ��ÿһ��
			{
				// �ж�����ͼ�Ƿ�Ϊ��ɫ���򡣽��Էǰ�ɫ���򣬲Ű�����ͷ�����ظ��ǵ�
				if (pimg[i * img1.cols * 3 + j * 3 + 0] + 
					pimg[i * img1.cols * 3 + j * 3 + 1] +
					pimg[i * img1.cols * 3 + j * 3 + 2] < 200 * 3)
				{
					// BGR�����̺죩����ͨ�����θ�ֵ
					pframe[i * img1.cols * 3 + j * 3 + 0] = pimg[i * img1.cols * 3 + j * 3 + 0];
					pframe[i * img1.cols * 3 + j * 3 + 1] = pimg[i * img1.cols * 3 + j * 3 + 1];
					pframe[i * img1.cols * 3 + j * 3 + 2] = pimg[i * img1.cols * 3 + j * 3 + 2];
				}
			}
		cv::imshow("image", frame); // ��ʾͼ��
		x += 3; // ����ͼÿ֡��������
		if (x > img.cols - win) // ������ͼ���ұ߽磬תͷ
			x = 0;
		cv::waitKey(1); // �ȴ�1���룬�Ա��ܹۿ���Ƶ
	}
	return 1;
}

