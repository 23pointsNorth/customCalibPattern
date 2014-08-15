#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

namespace cv{

class CustomPattern : public Algorithm
{
public:
	CustomPattern();
	~CustomPattern();
	bool create(InputArray pattern, const Size2f boardSize, OutputArray output = noArray());

	bool findPattern(InputArray image, OutputArray matched_features, OutputArray pattern_points, const double ratio = 0.7,
					 const double proj_error = 8.0, const bool refine_position = false, OutputArray out = noArray(),
					 OutputArray H = noArray(), OutputArray pattern_corners = noArray());
	/*
		accepting many images at the same time? FD::detect() does.
		matched_features -> vector<Point> is the feaures matched to
							the original set of points
		pattern_points -> the points from the original matched pattern
							(needed as not all points may be matched)
		refine_position -> refine point position thorough subPixelPos.
		H -> (Mat) The homography matrix.
		scene_corners -> std::vector<Point2f> vector of the 4 points describing the projected pattern corners
							(inside which the battern is bounded to)
		The two vectors can be used to calibrate the camera.
		@return: Found successfully.
	*/
	bool isInitialized();

	void getPatternPoints(OutputArray original_points);
	/*
		Returns a vector<Point> of the original points.
	*/
	double getPixelSize();
	/*
		Get the pixel size of the pattern
	*/

	bool setFeatureDetector(Ptr<FeatureDetector> featureDetector);
	bool setDescriptorExtractor(Ptr<DescriptorExtractor> extractor);
	bool setDescriptorMatcher(Ptr<DescriptorMatcher> matcher);

	Ptr<FeatureDetector> getFeatureDetector();
	Ptr<DescriptorExtractor> getDescriptorExtractor();
	Ptr<DescriptorMatcher> getDescriptorMatcher();

	double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
				Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
				OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = 0,
				TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));
	/*
		Calls the calirateCamera function with the same inputs.
	*/

	bool findRt(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs,
                OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int flags = ITERATIVE);
	bool findRt(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int flags = ITERATIVE);
	/*
		Uses solvePnP to find the rotation and translation of the pattern
		with respect to the camera frame.
	*/

	bool findRtRANSAC(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs,
				OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int iterationsCount = 100,
				float reprojectionError = 8.0, int minInliersCount = 100, OutputArray inliers = noArray(), int flags = ITERATIVE);
	bool findRtRANSAC(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
				OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int iterationsCount = 100,
				float reprojectionError = 8.0, int minInliersCount = 100, OutputArray inliers = noArray(), int flags = ITERATIVE);
    /*
		Uses solvePnPRansac()
	*/

	void drawOrientation(InputOutputArray image, InputArray tvec, InputArray rvec, InputArray cameraMatrix,
						 InputArray distCoeffs, double axis_length = 3, double axis_width = 2);
	/*
		pattern_corners -> projected over the image position of the edges of the pattern.
	*/

	/*
		Other:
		1. operator >> & <<; Be compatable with the save/load to XML/YAML.
		2. drawMatches(IOArray image);
		3. UpdateCameraCalibration(image, K, distCoeffs, *calib_params*);
	*/

private:
	//operator=

	Mat img_roi;
	std::vector<Point2f> obj_corners;
	double pxSize;

	bool initialized;

	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;

	std::vector<KeyPoint> keypoints;
	std::vector<Point3f> points3d;
	Mat descriptor;

	bool init(Mat& image, const float pixel_size, OutputArray output = noArray());
	bool findPatternPass(const Mat& image, std::vector<Point2f>& matched_features, std::vector<Point3f>& pattern_points,
						 Mat& H, std::vector<Point2f>& scene_corners, const double pratio, const double proj_error,
						 const bool refine_position = false, const Mat& mask = Mat(), OutputArray output = noArray());
	void scaleFoundPoints(const double squareSize, const std::vector<KeyPoint>& corners, std::vector<Point3f>& points3d);
	void check_matches(std::vector<Point2f>& matched, const std::vector<Point2f>& pattern, std::vector<DMatch>& good, std::vector<Point3f>& pattern_3d, const Mat& H);

	void keypoints2points(const std::vector<KeyPoint>& in, std::vector<Point2f>& out);
	void updateKeypointsPos(std::vector<KeyPoint>& in, const std::vector<Point2f>& new_pos);
	void refinePointsPos(const Mat& img, std::vector<Point2f>& p);
	void refineKeypointsPos(const Mat& img, std::vector<KeyPoint>& kp);
};

}
