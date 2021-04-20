#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include<sstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>

using namespace std;
using namespace cv;

stringstream ss;

#define BINARY_THRESHOLD 127
RNG rng(12345);

/*encodings :	0: skull
				1: foot
				2: hand
				3: knee
				4: hip
				*/



//sample encodings for grid size 20
//double encodings[5][4] = { { 0.209416,0.290584,0.209416,0.290584},{0.128731, 0.371269, 0.128731, 0.371269},{0.230623,0.269377, 0.230623, 0.269377},{ 0.142714, 0.357286, 0.142714, 0.357286},{0.267559,0.232441, 0.267559, 0.232441} };



//sample encodings for grid size 40
double encodings[5][4] = { {0.221773,0.278227, 0.221773, 0.278227},{ 0.142086, 0.357914, 0.142086, 0.357914},{0.222007, 0.277993, 0.222007, 0.277993},{0.130607,0.369393, 0.130607, 0.369393},{0.263647, 0.236353, 0.263647, 0.236353} };

//freeman code for the input image
double freecode[4] = { 0,0,0,0 };

//calculating freeman code from the object boundary
void freeman(Mat inputImage) {
	int flag = 0;
	int a, b, p = 0, q = 0;
	for (int i = 0; i < inputImage.rows; i++) {
		for (int j = 0; j < inputImage.cols; j++) {
			if (inputImage.at<uchar>(i, j) == 255) {
				flag = 1;
				p = i;
				q = j;
				break;
			}

		}
		if (flag == 1) {
			break;
		}
	}
	a = p;
	b = q;


	int x = 0;
	int flag2;
	int first = 0;
	int last = 0;
	do {
		flag2 = 0;
		if (x == 0) {
			if (last != 2 && q + 1 < inputImage.cols && inputImage.at<uchar>(p, q + 1) == 255) {
				freecode[0]++;
				flag2 = 1;
				q = q + 1;
				last = x;
			}
			else x++;
		}
		if (x == 1) {

			if (last != 3 && p + 1 < inputImage.rows && inputImage.at<uchar>(p + 1, q) == 255) {
				freecode[1]++;
				flag2 = 1;
				p = p + 1;
				last = x;
			}
			else x++;
		}
		if (x == 2) {
			if (last != 0 && q - 1 > -1 && inputImage.at<uchar>(p, q - 1) == 255) {
				freecode[2]++;
				flag2 = 1;
				q = q - 1;
				last = x;
			}
			else x++;
		}
		if (x == 3) {
			if (last != 1 && p - 1 > -1 && inputImage.at<uchar>(p - 1, q) == 255) {
				freecode[3]++;
				flag2 = 1;
				p = p - 1;
				last = x;
			}
			else x = 0;
		}
		if (first == 0 && flag2 == 1) {
			first = 1;
		}
	} while (first == 0 || !(p == a && q == b));

	return;

}



//getting the final object boundary for freeman coding
Mat getShape(Mat inputImage) {
	Mat grid = Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	int a, b, c, d;
	for (int i = 0; i < grid.rows; i++) {
		a = -1;
		b = -1;
		c = -1;
		d = -1;
		for (int y = 0; y < inputImage.cols; y++) {
			if (inputImage.at<uchar>(i, y) == 255) {
				a = i;
				b = y;
				break;
			}
		}
		if (a != -1) {
			for (int y = inputImage.cols - 1; y > b; y--) {
				if (inputImage.at<uchar>(i, y) == 255) {
					c = i;
					d = y;
					break;
				}
			}
		}
		if (c != -1) {
			cv::line(grid, Point(b, a), Point(d, c), cv::Scalar(255, 255, 255), 1);
		}
	}
	for (int j = 0; j < grid.cols; j++) {
		a = -1;
		b = -1;
		c = -1;
		d = -1;
		for (int x = 0; x < inputImage.rows; x++) {
			if (inputImage.at<uchar>(x, j) == 255) {
				a = x;
				b = j;
				break;
			}
		}
		if (a != -1) {
			for (int x = inputImage.rows - 1; x > a; x--) {
				if (inputImage.at<uchar>(x, j) == 255) {
					c = x;
					d = j;
					break;
				}
			}
		}
		if (c != -1) {
			cv::line(grid, Point(b, a), Point(d, c), cv::Scalar(255, 255, 255), 1);
		}
	}

	return grid;
}


//creating grid using inputImage grid nodes
Mat createGrid(Mat inputImage) {

	int dist = 40;
	Mat transformedImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC3);

	int width = inputImage.size().width;
	int height = inputImage.size().height;

	for (int i = 0; i < height; i += dist)
		cv::line(transformedImage, Point(i, 0), Point(i, width), cv::Scalar(255, 0, 0));

	for (int i = 0; i < width; i += dist)
		cv::line(transformedImage, Point(0, i), Point(height, i), cv::Scalar(255, 0, 0));


	return transformedImage;
}



//getting nearest grid node points from the object boundary
Mat gridTransform(Mat inputImage) {

	Mat outputImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);

	int p = 0, q = 0;
	int t = 40;					//grid size
	for (int i = 0; i < inputImage.rows; i++) {
		for (int j = 0; j < inputImage.cols; j++) {
			if (inputImage.at<uchar>(i, j) == 255) {

				p = i;
				q = j;

				p = p + (t / 2);
				p = p - (p % t);


				q = q + (t / 2);
				q = q - (q % t);


				if (p >= inputImage.rows) {
					p = p - t;
				}
				if (q >= inputImage.cols) {
					q = q - t;
				}

				outputImage.at<uchar>(p, q) = 255;

			}
		}
	}
	return outputImage;

}


//3x3 square structure for erosion and dilation
Mat callStructure(Mat inputImage, string flag)
{
	Mat outputImage = inputImage.clone();

	int intensity = 0;
	if (flag == "Dilate")
	{
		intensity = 255;
	}
	else if (flag == "Erode")
	{
		intensity = 0;
	}

	for (int i = 0; i < inputImage.rows; i++)
	{
		for (int j = 0; j < inputImage.cols; j++)
		{
			if (inputImage.at <uchar>(i, j) == intensity)
			{
				outputImage.at<uchar>(i, j) = intensity;
				if (i > 0)
					outputImage.at<uchar>(i - 1, j) = intensity;
				if (j > 0)
					outputImage.at<uchar>(i, j - 1) = intensity;
				if (i > 0 && j > 0)
					outputImage.at<uchar>(i - 1, j - 1) = intensity;
				if ((i + 1) < inputImage.rows)
					outputImage.at<uchar>(i + 1, j) = intensity;
				if ((j + 1) < inputImage.cols)
					outputImage.at<uchar>(i, j + 1) = intensity;
				if ((i + 1) < inputImage.rows && (j + 1) < inputImage.cols)
					outputImage.at<uchar>(i + 1, j + 1) = intensity;
				if (i > 0 && (j + 1) < inputImage.cols)
					outputImage.at<uchar>(i - 1, j + 1) = intensity;
				if ((i + 1) < inputImage.rows && j > 0)
					outputImage.at<uchar>(i + 1, j - 1) = intensity;
			}
		}
	}
	return outputImage;
}


//creating binary image from input image
Mat createBinaryImage(Mat inputimage)
{
	Mat binaryImage = inputimage.clone();

	for (int i = 0; i < inputimage.rows; i++)
	{
		for (int j = 0; j < inputimage.cols; j++)
		{
			if (inputimage.at<uchar>(i, j) >= BINARY_THRESHOLD)
				binaryImage.at<uchar>(i, j) = saturate_cast<uchar>(255);
			else
				binaryImage.at<uchar>(i, j) = saturate_cast<uchar>(0);
		}
	}
	return binaryImage;
}


//reading input image file
Mat readImage(string& fileName, string type)
{

	/*cin >> fileName;*/
	//fileName = "skull1.jpg";
	//fileName = "skull2.jpg";

	cout << "File Selected: " << fileName << endl;
	Mat inputImage = imread(fileName, 0);

	return inputImage;
}

int main() {

	cout << "Classification of X-ray Images Using Grid Approach" << endl;


	//all 25 input image names stored in array 
	string inputfiles[25] = { "foot1.jpg","foot2.jpg","foot3.jpg","foot4.jpg","foot5.jpg","hand1.jpg","hand2.jpg","hand3.jpg","hand4.jpg","hand5.jpg","hip1.jpg","hip2.jpg","hip3.jpg","hip4.jpeg","hip5.JPG", "knee1.jpg","knee2.jpg","knee3.jpg","knee4.jpg","knee5.png","skull1.jpg","skull2.jpg","skull3.jpg","skull4.jpg","skull5.jpg" };

	for (int i = 0; i < 25; i++) {
		cout << i+1 << ": " << inputfiles[i] << endl;
	}

	char q;

	do {
		int n;
		cout << "enter file number (1-25) :";
		cin >> n;


		//reading input image
		string fileName = inputfiles[n - 1];

		Mat inputImage = readImage(fileName, "Input");

		if (inputImage.empty()) {
			cerr << "Error: Loading image" << endl;
			char c = getchar();
			return -1;
		}


		//converting into binary
		Mat binaryImage = createBinaryImage(inputImage);
		Mat outputImage = binaryImage;

		//smoothing the binary image
		blur(binaryImage, outputImage, Size(3, 3));

		//getting canny edge
		Mat canny;
		Canny(outputImage, canny, 30, 100);

		//dilating the canny edge with 9x9 square structuring element
		Mat dilatedImage = canny.clone();
		for (int k = 0; k < 4; k++) {
			dilatedImage = callStructure(canny, "Dilate");
			canny = dilatedImage;
		}


		//floodfilling the dilated edge from both sides of the image
		Mat im_floodfill = dilatedImage.clone();
		floodFill(im_floodfill, cv::Point(0, 0), Scalar(255));
		floodFill(im_floodfill, cv::Point(im_floodfill.cols - 1, 0), Scalar(255));

		// Inverting the floodfilled image
		Mat im_floodfill_inv;

		bitwise_not(im_floodfill, im_floodfill_inv);



		// Combining the two images to get the foreground.

		Mat im_out = (dilatedImage | im_floodfill_inv);


		// eroding the foreground using 9x9 square structuring element 
		Mat erodedImage = im_out.clone();
		for (int k = 0; k < 4; k++) {
			erodedImage = callStructure(erodedImage, "Erode");
		}

		//Canny edge of eroded foreground
		Mat canny2;
		Canny(erodedImage, canny2, 30, 100);
		canny2 = callStructure(canny2, "Dilate");

		//getting the grid nodes from canny edge 
		Mat transformedImage = Mat::zeros(canny2.rows, canny2.cols, CV_8UC1);

		transformedImage = gridTransform(canny2);

		//finding the first 255 pixel
		Mat gridImage = transformedImage.clone();
		int i = 0, j = 0;
		int flag = 0;
		for (i = 0; i < gridImage.rows; i++) {
			for (j = 0; j < gridImage.cols; j++) {
				if (gridImage.at<uchar>(i, j) == 255) {
					flag = 1;
					break;
				}
			}
			if (flag == 1) break;
		}

		//joining grid nodes to get the grid transformed object boundary
		Mat output = getShape(transformedImage);

		Mat im_floodfill2 = output.clone();
		floodFill(im_floodfill2, cv::Point(0, 0), Scalar(255));
		floodFill(im_floodfill2, cv::Point(im_floodfill2.cols - 1, 0), Scalar(255));

		// Invert floodfilled image
		Mat im_floodfill_inv2;

		bitwise_not(im_floodfill2, im_floodfill_inv2);



		// Combine the two images to get the foreground.

		Mat im_out2 = (output | im_floodfill_inv2);
		im_out2 = callStructure(im_out2, "Erode");


		//getting the final transformed edge 
		Mat edge_image, eroded_image;
		cv::erode(im_out2, eroded_image, cv::Mat());

		edge_image = im_out2 - eroded_image;


		//calculating 4 dimensional freeman code 
		freeman(edge_image);

		//converting into direction probability
		double sum = freecode[0] + freecode[1] + freecode[2] + freecode[3];

		for (int i = 0; i < 4; i++) {
			freecode[i] /= sum;
		}

		//displaying normalised freeman code
		cout << "freeman code : " << freecode[0] << " " << freecode[1] << " " << freecode[2] << " " << freecode[3] << endl;


		namedWindow("Input Image", WINDOW_NORMAL);
		resizeWindow("Input Image", 600, 600);
		imshow("Input Image", inputImage);

		namedWindow("binaryImage", WINDOW_NORMAL);
		resizeWindow("binaryImage", 600, 600);
		imshow("binaryImage", outputImage);

		namedWindow("Canny Edge", WINDOW_NORMAL);
		resizeWindow("Canny Edge", 600, 600);
		imshow("Canny Edge", canny);

		namedWindow("Dilated Image", WINDOW_NORMAL);
		resizeWindow("Dilated Image", 600, 600);
		imshow("Dilated Image", dilatedImage);

		namedWindow("Floodfilled Image", WINDOW_NORMAL);
		namedWindow("Inverted Floodfilled Image", WINDOW_NORMAL);
		namedWindow("Foreground", WINDOW_NORMAL);

		resizeWindow("Floodfilled Image", 600, 600);
		resizeWindow("Inverted Floodfilled Image", 600, 600);
		resizeWindow("Foreground", 600, 600);

		imshow("Floodfilled Image", im_floodfill);
		imshow("Inverted Floodfilled Image", im_floodfill_inv);
		imshow("Foreground", im_out);

		namedWindow("Eroded Image", WINDOW_NORMAL);
		resizeWindow("Eroded Image", 600, 600);
		imshow("Eroded Image", erodedImage);

		namedWindow("Canny Edge 2", WINDOW_NORMAL);
		resizeWindow("Canny Edge 2", 600, 600);
		imshow("Canny Edge 2", canny2);

		namedWindow("Transform", WINDOW_NORMAL);
		resizeWindow("Transform", 600, 600);
		imshow("Transform", transformedImage);

		namedWindow("Grid Output", WINDOW_NORMAL);
		resizeWindow("Grid Output", 600, 600);
		imshow("Grid Output", output);

		namedWindow("Floodfilled Image 2", WINDOW_NORMAL);
		namedWindow("Inverted Floodfilled Image 2", WINDOW_NORMAL);
		namedWindow("Foreground 2", WINDOW_NORMAL);

		resizeWindow("Floodfilled Image 2", 600, 600);
		resizeWindow("Inverted Floodfilled Image 2", 600, 600);
		resizeWindow("Foreground 2", 600, 600);

		imshow("Floodfilled Image 2", im_floodfill2);
		imshow("Inverted Floodfilled Image 2", im_floodfill_inv2);
		imshow("Foreground 2", im_out2);

		namedWindow("Edge", WINDOW_NORMAL);
		resizeWindow("Edge", 600, 600);
		imshow("Edge", edge_image);

		waitKey();
		cout << "continue? (y/n):";
		cin >> q;
	}while (q == 'y');

	return 0;

}
