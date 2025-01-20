#include <opencv2/opencv.hpp>

using namespace cv;

int main(void) {
	std::string project_folder = "./SampleCode/";
  
	// Read image in GrayScale mode
	Mat image = imread(project_folder + "boy.jpg",0);

	// Save grayscale image
	imwrite(project_folder + "boyGray.jpg",image);

	return 0;
}
