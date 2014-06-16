#ifndef CUSTOM_PATTERN
#define CUSTOM_PATTERN

#include "custom_pattern.hpp"
#include <opencv2/opencv.hpp>

#include <vector>
#include <cstring>

using namespace std;

namespace cv{

CustomPattern::CustomPattern(InputArray image, const Rect roi,
						     const int flag, const Size patternSize, const float size,
						     OutputArray output)
{
    CV_Assert(!image.empty() && (roi.area() != 0) &&
              (patternSize.area() != 0) && (size != 0));
    CV_Assert((flag == CHESSBOARD_PATTERN)||(flag == CIRCLE_PATTERN));

    Mat img = image.getMat();

    vector<Point2f> corners;
    bool patternfound;
    // if(flag == CHESSBOARD_PATTERN)
    // {
    //     // CHESSBOARD_PATTERN
    //     patternfound = findChessboardCorners(image, patternSize, corners,
    //         CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
    // }
    // else
    // {
    //     // CIRCLE_PATTERN
    //     patternfound = findCirclesGrid(image, patternSize, corners);
    // }
    // Mat img_patterns(img);
    // drawChessboardCorners(img_patterns, patternSize, Mat(corners), patternfound);
    // imshow("Chessboard", img_patterns);

    if(patternfound || true /*for testing*/)
    {
        // Mat gray;
        // cvtColor(img, gray, COLOR_RGB2GRAY);
        // cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
        //             TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1));

        // // Average pixel size across the whole width.
        // Point2d box_len = (corners[0] - corners[patternSize.width]) * (1.0 / patternSize.width);
        double pixelSize = 1;//norm(box_len)/size;

        img(roi).copyTo(img_roi);

        detector = FeatureDetector::create("ORB");
        detector->set("nFeatures", 200);
        descriptorExtractor = DescriptorExtractor::create("ORB");

        detector->detect(img_roi, keypoints);
        cout << "Keypoints count: " << keypoints.size() << endl;
        descriptorExtractor->compute(img_roi, keypoints, descriptor);
        descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming(2)");
        descriptorMatcher->set("crossCheck", true);

        Mat o;
        drawKeypoints(img_roi, keypoints, o, CV_RGB(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Scale found points by pixelSize
        scaleFoundPoints(pixelSize, keypoints, points3d);

        initialized = (keypoints.size() != 0); // initialized if any keypoints are found
        if (output.needed()) o.copyTo(output);
    }
    else
    {
        //else no pattern found!
        initialized = false; // otherwise, give a warning, not error.
        CV_Error(Error::StsBadArg, "Could not find suggested pattern in the input image.");
    }
}

CustomPattern::~CustomPattern() {}

void CustomPattern::scaleFoundPoints(const double pixelSize,
            const vector<KeyPoint>& corners, vector<Point3f>& points3d)
{
    for (unsigned int i = 0; i < corners.size(); ++i)
    {
        points3d.push_back(Point3f(
                corners[i].pt.x * pixelSize,
                corners[i].pt.y * pixelSize,
                0));
    }
}

bool CustomPattern::findPattern(InputArray image, OutputArray matched_features,
                                                  OutputArray pattern_points)
{
    if (!initialized) {return false; }

    vector<DMatch> matches;
    vector<KeyPoint> f_keypoints;
    Mat f_descriptor;
    Mat img = image.getMat();

    detector->detect(img, f_keypoints);
    descriptorExtractor->compute(img, f_keypoints, f_descriptor);

    if(f_descriptor.type() != CV_32F) { f_descriptor.convertTo(f_descriptor, CV_32F);}
    if(descriptor.type() != CV_32F) { descriptor.convertTo(descriptor, CV_32F);}

    // cout << "Matching..." << endl;

    matcher.match(f_descriptor, descriptor, matches);
    // descriptorMatcher->match(f_descriptor, descriptor, matches);
    // Calculation of max and min distances between keypoints
    double max_dist = 0;
    double min_dist = 1e10;
    for(int i = 0; i < f_descriptor.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    // cout << "Max dist: " << max_dist << endl;
    // cout << "Min dist: " << min_dist << endl;
    // cout << "Keypnts1: " << keypoints.size() << endl;
    // cout << "Keypnts2: " << f_keypoints.size() << endl;

    vector<DMatch> good_matches;
    vector<Point3f> matched_3d_keypoints;
    vector<Point2f> matched_f_points, obj_points;

    for(int i = 0; i < f_descriptor.rows; i++)
    {
        if(matches[i].distance <= max(1.8 * min_dist, 0.02))
        {
            good_matches.push_back(matches[i]);
            // cout << "Adding point " << i << " with Qidx: " << matches[i].queryIdx << " Tidx: "  <<  matches[i].trainIdx << endl;
            // Collocate needed data for return
            // "keypoints1[matches[i].queryIdx] has a corresponding point in keypoints2[matches[i].trainIdx]"
            matched_f_points.push_back(f_keypoints[matches[i].queryIdx].pt);
            matched_3d_keypoints.push_back(points3d[matches[i].trainIdx]);
            // cout << "Point added." << endl;
            obj_points.push_back(keypoints[matches[i].trainIdx].pt);
        }
    }

    // cout << "Matched size: " << good_matches.size() << endl;

    Mat(matched_f_points).copyTo(matched_features);
    Mat(matched_3d_keypoints).copyTo(pattern_points);
    if (good_matches.size() < 4) return false;

    double max_error = 2;
    Mat H = findHomography(obj_points, matched_f_points, RANSAC, max_error);
    if (H.empty())
    {
        cout << "H: " << H.size() << endl;
        cout << "findHomography returned empty Mat." << endl;
        return false;
    }
    //-- Get the corners from the image
    vector<Point2f> obj_corners(4), scene_corners(4);
    obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(img_roi.cols, 0);
    obj_corners[2] = Point2f(img_roi.cols, img_roi.rows); obj_corners[3] = Point2f(0, img_roi.rows);

    perspectiveTransform(obj_corners, scene_corners, H);

    Mat img_matches(img.clone());
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
    imshow("lines", img_matches);
    waitKey(10);

    Mat out;
    drawMatches(img, f_keypoints, img_roi, keypoints, good_matches, out);
    // drawKeypoints(img, f_keypoints, img, CV_RGB(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Matched", out);
    return (!good_matches.empty()); // return true if there are enough good matches
}

void CustomPattern::getPatternPoints(OutputArray original_points)
{
    return Mat(keypoints).copyTo(original_points);
}

double CustomPattern::calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags,
                TermCriteria criteria)
{
    return calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria);
}


} // namespace cv



#endif