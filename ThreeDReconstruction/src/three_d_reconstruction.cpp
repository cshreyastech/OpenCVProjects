// https://stackoverflow.com/questions/7705377/3d-reconstruction-how-to-create-3d-model-from-2d-image
// https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
// https://amroamroamro.github.io/mexopencv/matlab/cv.stereoRectify.html


// #include "pch/pch.h"
#include <opencv2/opencv.hpp>
#include "opencv2/calib3d/calib3d.hpp"
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

int main()
{
  cv::Mat imageLeft = cv::imread("left_image.jpg");
  cv::Mat imageRight = cv::imread("right_image.jpg");

  
  cv::Mat rectifiedLeft, rectifiedRight;
  // Rectify images
  cv::Mat intrinsic_left    = cv::Mat(3, 3, CV_64F, 0.0);
  cv::Mat intrinsic_right   = cv::Mat(3, 3, CV_64F, 0.0);
  cv::Mat dist_coeffs_left  = cv::Mat(1, 5, CV_64F, 0.0);
  cv::Mat dist_coeffs_right = cv::Mat(1, 5, CV_64F, 0.0);
  const cv::Size image_size = imageLeft.size();

  // Output
  cv::Mat R1            = cv::Mat(3, 3, CV_64F, 0.0);
  cv::Mat R2            = cv::Mat(3, 3, CV_64F, 0.0);
  cv::Mat P1            = cv::Mat(3, 4, CV_64F, 0.0);
  cv::Mat P2            = cv::Mat(3, 4, CV_64F, 0.0);

  cv::Mat Q             = cv::Mat(4, 4, CV_64F, 0.0);

  const double fvg = 1.2;
  const double width = 640.0;
  const double height = 480.0;
  const double f = height / (2.0 * tan(fvg / 2.0));

  // Intrinsic calculations
  intrinsic_left.at<double>(0, 0) = f;
  intrinsic_left.at<double>(1, 1) = f;
  intrinsic_left.at<double>(0, 2) = width / 2.0;
  intrinsic_left.at<double>(1, 2) = height / 2.0;
  intrinsic_left.at<double>(2, 2) = 1.0f;
  intrinsic_right = intrinsic_left;

  std::cout << "intrinsic_left: \n"  << intrinsic_left << std::endl;
  std::cout << "intrinsic_right: \n" << intrinsic_right << std::endl;

  // Extrinsic calculations
  cv::Mat T_W_CL      = cv::Mat::zeros(4, 4, CV_64F);
  cv::Mat T_W_CR      = cv::Mat::zeros(4, 4, CV_64F);  

  T_W_CL.at<double>(0, 1) = 1.0;
  T_W_CL.at<double>(0, 3) = -0.02;
  T_W_CL.at<double>(1, 2) = 1.0;
  T_W_CL.at<double>(2, 0) = 1.0;
  T_W_CL.at<double>(2, 3) = -0.05;
  T_W_CL.at<double>(3, 3) = 1.0;

  T_W_CR.at<double>(0, 1) = 1.0;
  T_W_CR.at<double>(0, 3) = 0.02;
  T_W_CR.at<double>(1, 2) = 1.0;
  T_W_CR.at<double>(2, 0) = 1.0;
  T_W_CR.at<double>(2, 3) = -0.05;
  T_W_CR.at<double>(3, 3) = 1.0;

  cv::Mat R_CL_W = T_W_CL(cv::Range(0, 3), cv::Range(0, 3)).t();
  cv::Mat P_CL_W = -R_CL_W * T_W_CL(cv::Range(0, 3), cv::Range(3, 4));
  cv::Mat T_CL_W = cv::Mat::eye(4, 4, CV_64F);
  
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      T_CL_W.at<double>(i, j) = R_CL_W.at<double>(i, j);
  
  for(int i = 0; i < 3; i++)
    T_CL_W.at<double>(i, 3) = P_CL_W.at<double>(i);

  cv::Mat T_CL_CR = T_CL_W * T_W_CR;
  // std::cout << "T_CL_CR\n" << T_CL_CR << std::endl;

  cv::Mat R = T_CL_CR(cv::Range(0, 3), cv::Range(0, 3));
  cv::Mat T = T_CL_CR(cv::Range(0, 3), cv::Range(3, 4));
  
  // hardcode for testing
  T.at<double>(0) = 0.04;
  T.at<double>(1) = 0.0;
  // std::cout << "intrinsic_left: \n"   << intrinsic_left << std::endl;
  // std::cout << "dist_coeffs_left: \n" << dist_coeffs_left << std::endl;
  
  // std::cout << "intrinsic_right: \n"   << intrinsic_right << std::endl;
  // std::cout << "dist_coeffs_right: \n" << dist_coeffs_right << std::endl;
  
  // std::cout << "R: \n" << R << std::endl;
  std::cout << "T: \n" << T << std::endl;

// stereoRectify( InputArray cameraMatrix1, InputArray distCoeffs1,
//                                  InputArray cameraMatrix2, InputArray distCoeffs2,
//                                  Size imageSize, InputArray R, InputArray T,
//                                  OutputArray R1, OutputArray R2,
//                                  OutputArray P1, OutputArray P2,
//                                  OutputArray Q, int flags = CALIB_ZERO_DISPARITY,
//                                  double alpha = -1, Size newImageSize = Size(),
//                                  CV_OUT Rect* validPixROI1 = 0, CV_OUT Rect* validPixROI2 = 0 );


  cv::stereoRectify(intrinsic_left, dist_coeffs_left, intrinsic_right, 
    dist_coeffs_right, image_size, R, T, R1, R2, P1, P2, Q, 0);

  // P1 = cv::Mat(3, 4, CV_64F, 0.0);
  // P1.at<double>(2, 2) = 1;

  // P2 = cv::Mat(3, 4, CV_64F, 0.0);
  // P2.at<double>(2, 2) = 1;

  std::cout << "R1: \n" << R1 << std::endl;
  std::cout << "P1: \n" << P1 << std::endl;
  std::cout << "R2: \n" << R2 << std::endl;
  std::cout << "P2: \n" << P2 << std::endl;

  // cv::Mat rectMapLeft1, rectMapLeft2, rectMapRight1, rectMapRight2;

  // cv::initUndistortRectifyMap(intrinsic_left, dist_coeffs_left, R1, 
  //   P1, image_size, CV_32FC1, rectMapLeft1, rectMapLeft2);

  // std::cout << "rectifiedLeft: \n" << rectifiedLeft << std::endl;
  // std::cout << "rectMapLeft1: \n" << rectMapLeft1 << std::endl;
  // std::cout << "rectMapLeft2: \n" << rectMapLeft2 << std::endl;
  
  
  
  // cv::initUndistortRectifyMap(intrinsic_right, dist_coeffs_right, R2, 
  //   P2, image_size, CV_32FC1, rectMapRight1, rectMapRight2);


  // cv::remap(imageLeft, rectifiedLeft, rectMapLeft1, rectMapLeft2, cv::INTER_LINEAR);
  // cv::remap(imageRight, rectifiedRight, rectMapRight1, rectMapRight2, cv::INTER_LINEAR);

  // Now, rectifiedLeft and rectifiedRight are the rectified stereo images
  // Proceed with stereo matching and 3D reconstruction using these rectified images

  return 0;
}