#include "custom_pattern.hpp"

#include <iostream>

using namespace cv;
using namespace std;

#define	SQUARE_SIZE_M 0.035

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
    	// cout << "EVENT_LBUTTONDOWN: " << x  << " " << y <<endl;
    	mdown = true;
    }
    if(event == EVENT_LBUTTONUP)
    {
    	roi->width = x - roi->x;
    	roi->height = y - roi->y;
    	// cout << "EVENT_LBUTTONUP: " << x  << " " << y <<endl;
    	cout << "ROI: " << *roi << endl;
    	mdown = false;
    }
    if((event == EVENT_MOUSEMOVE) && mdown)
    {
    	roi->width = x - roi->x;
    	roi->height = y - roi->y;
    }
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
		vector<Point2f> matched;
		cout << "Calling" << endl;
		if (pattern->findPattern(frame, matched, org) && matched.size() > 3)
		{
			obj_points.push_back(org);
			matched_points.push_back(matched);
			cout << "Images collected: " << obj_points.size() << endl;
		}
		cout << "Called." << endl;

		imshow("Frame", frame);
		key = waitKey(10);
	}while(key != 'q');



	return 0;
}