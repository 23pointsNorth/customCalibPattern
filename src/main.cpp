#include "custom_pattern.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define	SQUARE_SIZE_M 0.024 //m

double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix , const Mat& distCoeffs,
                                         vector<float>& perViewErrors)
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); ++i )
    {
        projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                       distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);

        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

void onMouse(int event, int x, int y, int, void* data)
{
	static bool mdown = false;
	Rect* roi = (Rect*) data;

    if(event == EVENT_LBUTTONDOWN)
    {
    	roi->x = x;
    	roi->y = y;
    	roi->width = 0;
    	roi->height = 0;
    	mdown = true;
    }
    if(event == EVENT_LBUTTONUP)
    {
    	roi->width = x - roi->x;
    	roi->height = y - roi->y;
    	cout << "ROI: " << *roi << endl;
    	mdown = false;
    }
    if((event == EVENT_MOUSEMOVE) && mdown)
    {
    	roi->width = x - roi->x;
    	roi->height = y - roi->y;
    }
}

double total_mean = 0;
int tcount = 0;

void chessboard_accuracy(Mat& image)
{
	vector<Point2f> corners;
	Size boardSize(9, 6);
	bool patternfound = findChessboardCorners(image, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

	if (!patternfound) return;
#if 1
	Mat gray;
	cvtColor(image, gray, COLOR_RGB2GRAY);
	cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                    TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1));
#endif
	// create std points
	int squareSize = 10;
	vector<Point2f> board;
	 for( int i = 0; i < boardSize.height; ++i )
            for( int j = 0; j < boardSize.width; ++j )
                board.push_back(Point2f(float( j*squareSize ), float( i*squareSize )));


	Mat H = findHomography(board, corners, RANSAC);
    if (H.empty())
    {
        cout << "findHomography() returned empty Mat." << endl;
        return;
    }

    vector<Point2f> proj;
    perspectiveTransform(board, proj, H);

    double sum = 0;
    for (uint i = 0; i < proj.size(); ++i)
    {
    	double error = norm(proj[i] - corners[i]);
    	cout << error << endl;
    	sum += error;
    }

    cout << "Total error: " << sum << " over " << board.size() << " points." << endl;
    cout << "Mean error: " << sum / board.size() << endl;

    cout << "-----------" << endl;
    total_mean += (sum/board.size());
    tcount++;
    cout << "Total mean:" << total_mean / tcount << endl;
    cout << "-----------" << endl;
}

int main()
{
	CustomPattern* pattern;
	VideoCapture video(0);
	Rect roi;
	Mat frame;
	namedWindow("Select Pattern");
    setMouseCallback("Select Pattern", onMouse, &roi);

    char key;
	do
	{
		video >> frame;
		Mat canvas(frame.clone());
		rectangle(canvas, roi, CV_RGB(255, 0, 0));
		imshow("Select Pattern", canvas);
		key = waitKey(10);

		chessboard_accuracy(frame);

	}while(key != 't');
	destroyWindow("Select Pattern");

	cout << "ROI selected! Creating pattern!" << endl;
	Mat out;
	pattern = new CustomPattern(frame, roi, CHESSBOARD_PATTERN, Size(9, 6), SQUARE_SIZE_M, out);
	cout << "Pattern created." << endl;

	imshow("Algorithm", out);

	vector<vector<Point3f> > obj_points;
	vector<vector<Point2f> > matched_points;

	do
	{
		video >> frame;
		vector<Point3f> org;

		imshow("Frame", frame);
		key = waitKey(10);

		vector<Point2f> matched;
		// cout << "Calling" << endl;
		if (key == ' ' && pattern->findPattern(frame, matched, org) && matched.size() > 3)
		{
			obj_points.push_back(org);
			matched_points.push_back(matched);
			cout << "Matched size: " << matched.size() << " Images collected: " << obj_points.size() << endl;
		}
		// cout << "Called." << endl;
	}while(key != 'q');

	if (matched_points.empty()) return 0;

	Mat K, distCoeff;
	vector<Mat> rvec, tvec;
	cout << "RMS: " << pattern->calibrate(obj_points, matched_points, frame.size(), K, distCoeff, rvec, tvec) << endl;
	cout << "K: " << K << endl << "distCoeff" << distCoeff << endl;

	FileStorage fs("laptop_webcam_output.xml",  FileStorage::READ);
	fs["distortion_coefficients"] >> distCoeff;
	vector<float> perViewErrors;
	cout << "Alternatively Computed rms" << computeReprojectionErrors(obj_points,
                                         matched_points,
                                         rvec, tvec,
                                         K, distCoeff,
                                         perViewErrors) << endl;


	Mat undist;
	undistort(frame, undist, K, distCoeff);
	imwrite("undistorted.png", undist);
	imwrite("frame.png", frame);

	do
	{
		video >> frame;
		undistort(frame, undist, K, distCoeff);
		imshow("Undistorted", undist);
		key = waitKey(10);
	}while(key != 'q');

	return 0;
}



