#include <opencv2/opencv.hpp>

int main() {
    // Open the default camera (camera index 0)
    cv::VideoCapture cap(0);

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Get the default camera resolution
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Define the codec and create a VideoWriter object
    // Adjust the filename, codec, and parameters as needed
    cv::VideoWriter videoWriter("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(frameWidth, frameHeight));

    // Check if the VideoWriter object was initialized successfully
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not initialize VideoWriter." << std::endl;
        return -1;
    }

    // Loop to capture and record frames
    while (true) {
        cv::Mat frame;

        // Capture a frame from the camera
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Frame is empty." << std::endl;
            break;
        }

        // Display the frame (optional)
        cv::imshow("Camera", frame);

        // Write the frame to the video file
        videoWriter.write(frame);

        // Break the loop if the 'q' key is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the VideoCapture and VideoWriter objects
    cap.release();
    videoWriter.release();

    // Close all OpenCV windows
    cv::destroyAllWindows();

    return 0;
}
