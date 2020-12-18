#include <opencv2\core.hpp>
#include <opencv2\ml.hpp> 
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <iterator>  
#include <vector>  
using namespace std;
using namespace cv;
using namespace cv::ml;

void creatMat(Mat& x, Mat& y, String fileName) 
{
	int line_count = 0;
	char buffer[256];
	ifstream in(fileName);
	if (!in.is_open()) {
		cout << "Error opening file"; exit(1);
	}
	while (!in.eof())
	{
		in.getline(buffer, 100);
		stringstream stream;
		stream << buffer;
		string temp_s;
		stream >> temp_s;
		double num1 = atof(temp_s.c_str());
		if (num1 == 1.0) {
			y.at<double>(0, line_count) = num1;
		}
		while (stream >> temp_s) {
			int index = temp_s.find(':');
			string temp1_s = temp_s.substr(0, index);
			double t1 = atof(temp1_s.c_str());
			string temp2_s = temp_s.substr(index + 1, temp_s.length());
			double t2 = atof(temp2_s.c_str());
			//cout << t1 << "::"<<t2<<" ";

			x.at<double>(t1 - 1, line_count) = t2;
		}
		line_count++;
	}
}

void sigmoid(const Mat& z, Mat& a) {
	double temp;
	for (int i = 0; i < z.rows; i++) {
		for (int j = 0; j < z.cols; j++) {
			temp = z.at<double>(i, j);
			a.at<double>(i, j) = 1.0 / (1.0 + exp(-temp));
		}
	}
}

void change_log(const Mat& origin, Mat& response) {
	double temp;
	for (int i = 0; i < origin.rows; i++) {
		for (int j = 0; j < origin.cols; j++) {
			temp = origin.at<double>(i, j);
			response.at<double>(i, j) = log(temp);
		}
	}
}

float compute_cost(const Mat& y, const Mat& a) {
	double cost = 0.0;
	cv::Mat temp1 = cv::Mat::zeros(a.rows, a.cols, CV_64FC1);
	cv::Mat temp2 = cv::Mat::zeros(a.rows, a.cols, CV_64FC1);
	change_log(a, temp1);
	change_log(1 - a, temp2);
	cv::Mat loss;
	loss = y.mul(temp1) + (1 - y).mul(temp2);
	cost = (-1.0 / y.cols) * sum(loss)[0];
	return cost;
}

double propagate(Mat& w, double& b, const Mat& x, const Mat& y, Mat& a, Mat& dw, double& db) {
	cv::Mat z;
	z = w.t() * x + b;
	sigmoid(z, a);
	double cost = compute_cost(y, a);
	dw = (1.0 / y.cols) * (x * (a - y).t());
	db = (1.0 / y.cols) * sum(a - y)[0];
	return cost;
}

//计算分类精度
float calculateAccuracyPercent(const Mat& original, const Mat& predicted)
{
	return 100 * (float)countNonZero(original == predicted) / predicted.cols;
}
int main()
{
	cv::Mat train_x = cv::Mat::zeros(123, 1605, CV_64FC1); // 全零矩阵
	cv::Mat train_y = cv::Mat::zeros(1, 1605, CV_64FC1); // 全零矩阵
	string filename1 = "D:\\桌面转移\\VS C++ Code\\genetic algorithm\\a.txt";
	creatMat(train_x, train_y, filename1);


	cv::Mat test_x = cv::Mat::zeros(123, 30956, CV_64FC1); // 全零矩阵
	cv::Mat test_y = cv::Mat::zeros(1, 30956, CV_64FC1); // 全零矩阵
	string filename2 = "D:\\桌面转移\\VS C++ Code\\genetic algorithm\\test.txt";
	creatMat(test_x, test_y, filename2);
	
	//初始化W
	cv::Mat w = cv::Mat::zeros(123, 1, CV_64FC1); // 全零矩阵
	//初始化b
	double b = 0;

	cv::Mat a = cv::Mat::zeros(1, 1605, CV_64FC1);


	int num_iterations = 10000;//迭代次数
	double learning_rate = 0.009;//学习速率
	for (int i = 0; i < num_iterations; i++) {
		cv::Mat dw;
		double db;
		double cost = propagate(w, b, train_x, train_y, a, dw, db);
		//cout << dw;
		w = w - learning_rate * dw;
		b = b - learning_rate * db;
		if (i % 100 == 0) {
			cout << (i / 100) + 1 << ":" << cost << endl;
		}
	}
	/*for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			if (a.at<double>(i, j) >= 0.5) {
				a.at<double>(i, j) = 1.0;
			}
			else {
				a.at<double>(i, j) = 0.0;
			}
		}
	}*/

	cv::Mat a_test = cv::Mat::zeros(1, 30956, CV_64FC1);
	cv::Mat dw;
	double db;
	propagate(w, b, test_x, test_y, a_test, dw, db);
	//cout << a;
	train_y.convertTo(train_y, CV_32S);  //转换为整型
	a.convertTo(a, CV_32S);  //转换为整型
	cout << "train accuracy: " << calculateAccuracyPercent(train_y, a) << "%" << endl;

	test_y.convertTo(test_y, CV_32S);  //转换为整型
	a_test.convertTo(a_test, CV_32S);  //转换为整型
	cout << "test accuracy: " << calculateAccuracyPercent(test_y, a_test) << "%" << endl;
	int temp;
	cin >> temp;
}