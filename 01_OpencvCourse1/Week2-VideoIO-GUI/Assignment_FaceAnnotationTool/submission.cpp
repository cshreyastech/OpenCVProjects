// Face Annotate Tool -Assignment
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

std::string DATA_PATH = 
  "./01_OpencvCourse1/Week2-VideoIO-GUI/Assignment_FaceAnnotationTool/";
std::string RESULTS_PATH = 
  "./01_OpencvCourse1/Week2-VideoIO-GUI/Assignment_FaceAnnotationTool/results/";

// Points to store the center of the circle and a point on the circumference
cv::Point start_point, end_point;
cv::Mat source_img, crop;
int mouse_action;

// function which will be called on mouse input
void AnnotateTool(int action, int x, int y, int flags, void *userdata)
{
  mouse_action = action;

  // Action to be taken when left mouse button is pressed
  if( action == cv::EVENT_LBUTTONDOWN)
  {
    start_point = cv::Point(x,y);
  }
  // Action to be taken when left mouse button is released
  else if( action == cv::EVENT_LBUTTONUP)
  {
    end_point = cv::Point(x,y);
    crop = source_img(cv::Range(start_point.y, end_point.y),
      cv::Range(start_point.x, end_point.x));
  }  
}

int main()
{
  // Image Path
  std::string imgPath = DATA_PATH + "images/boy.jpg";

  // Read image and store it in variable img
  source_img = cv::imread(imgPath, cv::IMREAD_COLOR);
  if(source_img.empty())
    throw std::runtime_error("Could not open or find the impage");

  // Make a dummy image, will be useful to clear the drawing
  cv::Mat dummy = source_img.clone();

  cv::namedWindow("Window");
  // highgui function called when mouse events occur
  cv::setMouseCallback("Window", AnnotateTool);

  int k = 0;
  // loop until escape character is pressed or mouse left button up
  while(k!=27 && mouse_action != cv::EVENT_LBUTTONUP)
  {
    cv::imshow("Window", source_img);
      putText(source_img, "Choose top left corner, and drag?",
      cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
      cv::Scalar(255,255,255), 2 );

    k = cv::waitKey(20) & 0xFF;

    if(k== 99)
      // Another way of cloning
      dummy.copyTo(source_img);
  }
  
  imwrite(RESULTS_PATH + "crop.png", crop);
  
  cv::destroyAllWindows();
  return 0;
}