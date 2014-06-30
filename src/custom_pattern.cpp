#ifndef CUSTOM_PATTERN
#define CUSTOM_PATTERN

#include "custom_pattern.hpp"
#include <opencv2/opencv.hpp>

#include <vector>
#include <cstring>

using namespace std;

#define FLANN_ON    0

#define MIN_CONTOUR_AREA_PX     100
#define MIN_CONTOUR_AREA_RATIO  0.2
#define MAX_CONTOUR_AREA_RATIO  5

#define MIN_POINTS_FOR_H        10

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
        detector->set("nFeatures", 2000);
        detector->set("scaleFactor", 1.15);
        detector->set("nLevels", 30);
        descriptorExtractor = DescriptorExtractor::create("ORB");

        detector->detect(img_roi, keypoints);
        cout << "Keypoints count: " << keypoints.size() << endl;
        descriptorExtractor->compute(img_roi, keypoints, descriptor);

        #if (not FLANN_ON)
        descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming(2)");
        // descriptorMatcher->set("crossCheck", true); // not valid with k!=1
        cout << "BruteForce-Hamming(2) matcher." << endl;
        #endif

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

template<typename Tstve>
void deleteStdVecElem(vector<Tstve>& v, int idx)
{
    v[idx] = v.back();
    v.pop_back();
}

bool CustomPattern::findPattern(InputArray image, OutputArray matched_features,
                                                  OutputArray pattern_points)
{
    if (!initialized) {return false; }

    vector<vector<DMatch> > matches;
    vector<KeyPoint> f_keypoints;
    Mat f_descriptor;
    Mat img = image.getMat();

    detector->detect(img, f_keypoints);
    descriptorExtractor->compute(img, f_keypoints, f_descriptor);

    // cout << "Matching..." << endl;

    #if FLANN_ON
    if(f_descriptor.type() != CV_32F) { f_descriptor.convertTo(f_descriptor, CV_32F);}
    if(descriptor.type() != CV_32F) { descriptor.convertTo(descriptor, CV_32F);}
    matcher.knnMatch(f_descriptor, descriptor, matches, 2); // k = 2;
    #else
    descriptorMatcher->knnMatch(f_descriptor, descriptor, matches, 2); // k = 2;
    #endif

    // cout << "Choosing best matches!" << endl;
    vector<DMatch> good_matches;
    vector<Point3f> matched_3d_keypoints;
    vector<Point2f> matched_f_points, obj_points;

    for(int i = 0; i < f_descriptor.rows; ++i)
    {
        if(matches[i][0].distance < 0.60 * matches[i][1].distance)
        {
            const DMatch& dm = matches[i][0];
            good_matches.push_back(dm);
            // cout << "Adding point " << i << " with Qidx: " << dm.queryIdx << " Tidx: "  <<  dm.trainIdx << endl;
            // Collocate needed data for return
            // "keypoints1[matches[i].queryIdx] has a corresponding point in keypoints2[matches[i].trainIdx]"
            matched_f_points.push_back(f_keypoints[dm.queryIdx].pt);
            matched_3d_keypoints.push_back(points3d[dm.trainIdx]);
            obj_points.push_back(keypoints[dm.trainIdx].pt);
            // cout << "Point added." << endl;
        }
    }
    cout << "After point ratio size: " << good_matches.size() << endl;

    if (good_matches.size() < MIN_POINTS_FOR_H) return false; // 2*3 + 1 = RANSAC 50%+1

    double max_error = 8;
    Mat mask; // or vector<uchar>
    Mat H = findHomography(obj_points, matched_f_points, RANSAC, max_error, mask);
    if (H.empty())
    {
        cout << "findHomography() returned empty Mat." << endl;
        return false;
    }

    for(unsigned int i = 0; i < good_matches.size(); ++i)
    {
        if(!mask.data[i])
        {
            deleteStdVecElem(good_matches, i);
            deleteStdVecElem(matched_f_points, i);
            deleteStdVecElem(matched_3d_keypoints, i);
        }
    }

    cout << "After findHomography: " << good_matches.size() << endl;
    if (good_matches.empty()) return false;


    // Get the corners from the image
    vector<Point2f> obj_corners(4), scene_corners(4);
    obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(img_roi.cols, 0);
    obj_corners[2] = Point2f(img_roi.cols, img_roi.rows); obj_corners[3] = Point2f(0, img_roi.rows);

    perspectiveTransform(obj_corners, scene_corners, H);
    // Mat img_matches(img.clone());
    // Draw lines between the corners (the mapped object in the scene - image_2 )

    // Check correctnes of H
    // Is is a convex hull?
    bool cConvex = isContourConvex(scene_corners);
    cout << "IsContourConvex -- " << cConvex << endl;
    cout << "Points are: " << scene_corners[0] << scene_corners[1] << scene_corners[2] << scene_corners[3] << endl;
    if (!cConvex) return false;

    // Is the hull too large or small?
    double scene_area = contourArea(scene_corners);
    cout << "Contour Area -- " << scene_area << endl;
    if (scene_area < MIN_CONTOUR_AREA_PX) return false;
    double ratio = scene_area/img_roi.size().area();
    cout << "Area ratio -- " << ratio << endl;
    if ((ratio < MIN_CONTOUR_AREA_RATIO) ||
        (ratio > MAX_CONTOUR_AREA_RATIO)) return false;


    // Is any of the projected points outside the hull?
    int k = 0;
    for(unsigned int i = 0; i < good_matches.size(); ++i)
    {
        if(pointPolygonTest(scene_corners, f_keypoints[good_matches[i].queryIdx].pt, false) < 0)
        {
            ++k;
            deleteStdVecElem(good_matches, i);
            deleteStdVecElem(matched_f_points, i);
            deleteStdVecElem(matched_3d_keypoints, i);
        }
    }
    cout << "K: " << k << endl;

    Mat out;
    // imshow("lines", out);
    drawMatches(img, f_keypoints, img_roi, keypoints, good_matches, out);
    line(out, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
    line(out, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
    line(out, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
    line(out, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);
    cout << "@@@@@@@@@@@@@@@@@@@@@@@@@ BIG: " << good_matches.size() << endl;
    imshow("Matched", out);
    waitKey(10);

    Mat(matched_f_points).copyTo(matched_features);
    Mat(matched_3d_keypoints).copyTo(pattern_points);

    // RE-Matching
    // Expand image if pattern roi outside
    Rect r = boundingRect(scene_corners);
    int left = 0, right = 0, top = 0, bottom = 0;
    if (r.x < 0) { left = abs(r.x); r += Point(left, 0); }
    if (r.y < 0) { top = abs(r.y); r += Point(0, top); }
    if (r.x + r.width > img.cols) { right = (r.x + r.width) - img.cols;}
    if (r.y + r.height > img.rows) { bottom = (r.y + r.height) - img.rows;}
    Mat img_big;
    copyMakeBorder(img, img_big, top, bottom, left, right, BORDER_CONSTANT, CV_RGB(255, 255, 255));

    // Detect
    Mat i = img_big(r);
    detector->detect(i, f_keypoints);
    // Mat o;
    // drawKeypoints(i, f_keypoints, o, CV_RGB(255, 0, 0));
    // imshow("re-matched keypoints", o);

    // Rematch and filter
    descriptorExtractor->compute(i, f_keypoints, f_descriptor);
    good_matches.clear();
    matches.clear();
    descriptorMatcher->knnMatch(f_descriptor, descriptor, matches, 2);
    for(int i = 0; i < f_descriptor.rows; ++i)
    {
        if(matches[i][0].distance < 0.75 * matches[i][1].distance)
        {
            const DMatch& dm = matches[i][0];
            good_matches.push_back(dm);
        }
    }
    drawMatches(i, f_keypoints, img_roi, keypoints, good_matches, out);
    cout << "@@@@@@@@@@@@@@@@@@@@@@@ SMALL: " << good_matches.size() << endl;
    imshow("RE-Matched", out);

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