#include <iostream>
#include <deque>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::VideoCapture vs;
    cv::Mat frame, blurred, hsv, mask;

    // construct the argument parse and parse the arguments
    cv::CommandLineParser parser(argc, argv);
    std::string videoPath = parser.get<std::string>("-v");
    int buffer_size = parser.get<int>("-b");

    // define the lower and upper boundaries of the "green" ball in the HSV color space
    cv::Scalar greenLower(100, 100, 100);
    cv::Scalar greenUpper(255, 255, 255);

    // initialize the list of tracked points, the frame counter, and the coordinate deltas
    std::deque<cv::Point> pts;
    int counter = 0;
    int dX = 0, dY = 0;

    std::string direction;

    // if a video path was not supplied, grab the reference to the webcam
    if (videoPath.empty()) {
        vs.open(0);
    }
    // otherwise, grab a reference to the video file
    else {
        vs.open(videoPath);
    }

    // allow the camera or video file to warm up
    cv::waitKey(2000);

    // keep looping
    while (true) {
        // grab the current frame
        vs.read(frame);

        // if we did not grab a frame, then we have reached the end of the video
        if (frame.empty()) {
            break;
        }

        // resize the frame, blur it, and convert it to the HSV color space
        cv::resize(frame, frame, cv::Size(600, 0));
        cv::GaussianBlur(frame, blurred, cv::Size(11, 11), 0);
        cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

        // construct a mask for the color "green", then perform a series of dilations and erosions
        mask = cv::inRange(hsv, greenLower, greenUpper);
        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);

        // find contours in the mask
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(mask.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::Point> center;

        // only proceed if at least one contour was found
        if (!cnts.empty()) {
            // find the largest contour
            size_t maxIndex = 0;
            double maxArea = cv::contourArea(cnts[0]);
            for (size_t i = 1; i < cnts.size(); ++i) {
                double area = cv::contourArea(cnts[i]);
                if (area > maxArea) {
                    maxArea = area;
                    maxIndex = i;
                }
            }

            // compute the minimum enclosing circle and centroid
            cv::minEnclosingCircle(cnts[maxIndex], center);
            cv::Moments M = cv::moments(cnts[maxIndex]);
            cv::Point centroid(int(M.m10 / M.m00), int(M.m01 / M.m00));

            // only proceed if the radius meets a minimum size
            if (maxArea > 10) {
                // draw the circle and centroid on the frame
                cv::circle(frame, center[0], int(maxArea), cv::Scalar(0, 255, 255), 2);
                cv::circle(frame, centroid, 5, cv::Scalar(0, 0, 255), -1);
                pts.push_front(centroid);
            }
        }

        // loop over the set of tracked points
        for (size_t i = 1; i < pts.size(); ++i) {
            // if either of the tracked points are None, ignore them
            if (pts[i - 1].x == 0 && pts[i - 1].y == 0 || pts[i].x == 0 && pts[i].y == 0) {
                continue;
            }

            // compute the thickness of the line and draw the connecting lines
            int thickness = int(std::sqrt(buffer_size / float(i + 1)) * 2.5);
            cv::line(frame, pts[i - 1], pts[i], cv::Scalar(0, 0, 255), thickness);
        }

        // show the frame
        cv::imshow("Frame", frame);

        // show the movement deltas and the direction of movement
        std::cout << "Direction: " << direction << "\n";
        std::cout << "dx: " << dX << ", dy: " << dY << "\n";

        // increment the frame counter
        ++counter;

        // if the 'q' key is pressed, stop the loop
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // release the camera
    vs.release();

    // close all windows
    cv::destroyAllWindows();

    return 0;
}
