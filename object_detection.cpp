#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <memory>
#include <vector>

std::string matToString(const cv::Mat& mat) {
    std::vector<uchar> buffer;
    cv::imencode(".jpg", mat, buffer);
    return std::string(buffer.begin(), buffer.end());
}

int main(int argc, char* argv[]) {
    // Check if the input buffer address is provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path> <buffer_address>" << std::endl;
        return -1;
    }

    // Parse the input buffer address
    uchar* bufferAddress = reinterpret_cast<uchar*>(std::stoull(argv[2], nullptr, 16));

    // Create a VideoCapture object
    std::unique_ptr<cv::VideoCapture> cap;
    if (argc > 1) {
        cap = std::make_unique<cv::VideoCapture>(argv[1]);
    }
    else {
        std::cerr << "Please provide a video file path as a command-line argument." << std::endl;
        return -1;
    }

    // Check if the video is opened successfully
    if (!cap->isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // Create a background subtractor object
    std::shared_ptr<cv::BackgroundSubtractorMOG2> bg_subtractor = cv::createBackgroundSubtractorMOG2();

    // Loop over the video frames
    cv::Mat frame, fg_mask, binary_mask;
    while (true) {
        *cap >> frame; // Read the frame
        if (frame.empty()) break;

        cv::Mat fg_mask;
        bg_subtractor->apply(frame, fg_mask);

        cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

        // Calculate the mean value of the foreground mask
        double mean_value = cv::mean(fg_mask)[0];

        // Use input memory address as buffer address for binary_mask
        binary_mask = cv::Mat(frame.rows, frame.cols, frame.type(), bufferAddress);

        // Threshold the foreground mask to create binary mask
        cv::threshold(fg_mask, binary_mask, mean_value, 255, cv::THRESH_BINARY);

        // Find contours in the binary mask
        std::vector<std::vector<cv::Point>>* contours = new std::vector<std::vector<cv::Point>>();
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary_mask, *contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Draw polygons around moving objects
        for (size_t i = 0; i < contours->size(); ++i) {
            double area = cv::contourArea((*contours)[i]);

            // Set a threshold for contour area to filter out small contours
            if (500.0 < area || area < 1500.0) {  // Adjust the threshold as needed
                std::vector<cv::Point>* approx = new std::vector<cv::Point>();
                cv::approxPolyDP((*contours)[i], *approx, 5, true); // Adjust epsilon value (5) as needed
                cv::polylines(frame, *approx, true, cv::Scalar(0, 0, 255), 1.5);



                // Clean up dynamic memory for the approx vector
                delete approx;
            }
        }

        // Clean up dynamic memory for contours
        delete contours;
        // Display the frame
        cv::imshow("Frame", frame);

        // Release the memory of Mat variables
        fg_mask.release();
        binary_mask.release();

        // Press 'q' to quit
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the VideoCapture object
    cap.release();

    // Close all windows
    cv::destroyAllWindows();

    return 0;
}
