// QR-Code-Assignment
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


int main(){
	std::string DATA_PATH = "./01_OpencvCourse1/Week1-GettingStarted/Assigment-QRcodeDetector/";
  std::string RESULTS_PATH = "./01_OpencvCourse1/Week1-GettingStarted/Assigment-QRcodeDetector/results/";
  // Image Path
	string imgPath = DATA_PATH + "images/IDCard-Satya.png";

	// Read image and store it in variable img
	///
	/// YOUR CODE HERE
	///
  Mat img = cv::imread(imgPath, IMREAD_COLOR);
	
	Mat bbox, rectifiedImage;

	// Create a QRCodeDetector Object
	// Variable name should be qrDecoder
	///
	/// YOUR CODE HERE
	///
  cv::QRCodeDetector qrDecoder;

	// Detect QR Code in the Image
	// Output should be stored in opencvData
	///
	/// YOUR CODE HERE
	///
  cv::String opencvData = qrDecoder.detectAndDecode(img, bbox, rectifiedImage);
	// Check if a QR Code has been detected
	if(opencvData.length()>0)
		cout << "QR Code Detected" << endl;
	else
		cout << "QR Code NOT Detected" << endl;
	
	int n = bbox.rows; 
  // imshow("img", img);
  // imshow("bbox", bbox);
  // imshow("rectifiedImage-orgional", rectifiedImage);

	// Draw the bounding box
	///
	/// YOUR CODE HERE
	///
  Mat img_result = img.clone();
  
  int rectangle_start_x = 22;
  int rectangle_start_y = 82;
  int rectangle_side = 104;

  rectangle(img_result, Point(rectangle_start_x, rectangle_start_y), 
    Point(rectangle_start_x + rectangle_side, rectangle_start_y + rectangle_side),
          Scalar(255, 0, 0), 3, LINE_8);

	// Since we have already detected and decoded the QR Code
	// using qrDecoder.detectAndDecode, we will directly
	// use the decoded text we obtained at that step (opencvData)

	cout << "QR Code Detected!" << endl;
	///
	/// YOUR CODE HERE
	///
	std::cout << opencvData << std::endl;

	// Write the result image
	string resultImagePath = RESULTS_PATH + "./QRCode-Output.png";

	///
	/// YOUR CODE HERE
	///
	imwrite(resultImagePath, img_result);

  imshow("Draw rectangle on image", img_result);
  waitKey(0);
  destroyAllWindows();
	return 0;
}