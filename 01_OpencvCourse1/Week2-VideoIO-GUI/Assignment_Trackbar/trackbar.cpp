#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


int maxScaleUp = 100;
int scaleFactor = 1;
int scaleType = 0;
int maxType = 1;

string windowName = "Resize Image";
string trackbarValue = "Scale";
string trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down";

void scaleImage(int, void*);

std::string DATA_PATH = 
  "./01_OpencvCourse1/Week2-VideoIO-GUI/Assignment_Trackbar/";

// load an image
Mat im;
int main()
{

  // load an image
  im = imread(DATA_PATH + "images/truth.png");


  // Create a window to display results
  namedWindow(windowName, WINDOW_AUTOSIZE);

  // Create Trackbars and associate a callback function
  createTrackbar(trackbarValue, windowName, &scaleFactor, maxScaleUp, scaleImage);
  createTrackbar(trackbarType, windowName, &scaleType, maxType, scaleImage);

  scaleImage(25,0);

  while (true){
    int c;
    c = waitKey(20);
    if (static_cast<char>(c) == 27)
      break;
  }

  return 0;
}

// Callback functions
void scaleImage(int, void*)
{
  double scaleFactorDouble = 0;

  if(scaleType == 0)
  {
    // Get the Scale factor from the trackbar
    scaleFactorDouble = 1 + scaleFactor/100.0;
  }
  else
  {
    scaleFactorDouble = 1 - scaleFactor/100.0;
  }

  
  if (scaleFactorDouble == 0){
    scaleFactorDouble = 1;
  }
  
  Mat scaledImage;
  
  // Resize the image
  resize(im, scaledImage, Size(), scaleFactorDouble, scaleFactorDouble, INTER_LINEAR);
  imshow(windowName, scaledImage);
}