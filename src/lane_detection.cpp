#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cv.h>
#include <math.h>
#include "utils.h"
#include <vector>
#include <sys/time.h>

#define LEFT 1
#define MIDDLE 2
#define RIGHT 3
#define NONE 4 

void crop(IplImage* src,  IplImage* dest, CvRect rect) {
    cvSetImageROI(src, rect); 
    cvCopy(src, dest); 
    cvResetImageROI(src); 
}

struct Lane {
	Lane(){}
	Lane(CvPoint a, CvPoint b, float angle): p0(a),p1(b),angle(angle) { }

	CvPoint p0, p1;
	float angle;
};

enum{
	LEFT_MAX_LINE_REJECT_DEGREES = -8, // in degrees
	LEFT_MIN_LINE_REJECT_DEGREES = -50, // in degrees
	RIGHT_MAX_LINE_REJECT_DEGREES = 50, // in degrees
	RIGHT_MIN_LINE_REJECT_DEGREES = 8, // in degrees
	
	CANNY_MIN_TRESHOLD = 120,	  // edge detector minimum hysteresis threshold
	CANNY_MAX_TRESHOLD = 200, // edge detector maximum hysteresis threshold

	HOUGH_TRESHOLD = 50,		// line approval vote threshold
	HOUGH_MIN_LINE_LENGTH = 15,	// remove lines shorter than this treshold
	HOUGH_MAX_LINE_GAP = 150,   // join lines to one with smaller than this gaps

	FRAME_WAIT_KEY = 1,
	CAR_MIDDLE = 1050,

	LANE_COUNT_THRESHOLD = 0,
};

int processLanes(CvSeq* lines, IplImage* edges, IplImage* temp_frame, IplImage* frame, int* result, int size, int &current) {

	// classify lines to left/right side
	std::vector<Lane> left, right;

	CvFont font, font1;
	cvInitFont(&font, CV_FONT_VECTOR0, 1, 1, 1, 2, 8);
	cvInitFont(&font1, CV_FONT_VECTOR0, 2, 2, 1, 5, 8);
	
	for(int i = 0; i < lines->total; i++ )
    {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
		int midx = (line[0].x + line[1].x) / 2;
		int dx = line[1].x - line[0].x;
		int dy = line[1].y - line[0].y;
		float angle = atan2f(dy, dx) * 180/CV_PI;
		
		if(midx < CAR_MIDDLE){
			if(angle > LEFT_MAX_LINE_REJECT_DEGREES || angle < LEFT_MIN_LINE_REJECT_DEGREES)
				continue;
			left.push_back(Lane(line[0], line[1], angle));
		}
		else if(midx >= CAR_MIDDLE){
			if(angle > RIGHT_MAX_LINE_REJECT_DEGREES || angle < RIGHT_MIN_LINE_REJECT_DEGREES)
				continue;
			right.push_back(Lane(line[0], line[1], angle));
		}
		
		/*
		// show lines angle
		char ee[100];
		sprintf(ee, "%d", (int)angle);
		cvPutText(temp_frame, ee, cvPoint(int((line[1].x+line[0].x)/2), int((line[1].y+line[0].y)/2)), &font, cvScalar(255, 178, 0));
		*/

   }

	
	// show Hough lines
	for	(int i=0; i<right.size(); i++) {
		CvPoint p0, p1;
		p0.x = right[i].p0.x;
		p0.y = right[i].p0.y + 1080*2.5/3.5;
		p1.x = right[i].p1.x;
		p1.y = right[i].p1.y + 1080*2.5/3.5;
		cvLine(frame, p0, p1, CV_RGB(0, 0, 255), 2);
	}

	for	(int i=0; i<left.size(); i++) {
		CvPoint p0, p1;
		p0.x = left[i].p0.x;
		p0.y = left[i].p0.y + 1080*2.5/3.5;
		p1.x = left[i].p1.x;
		p1.y = left[i].p1.y + 1080*2.5/3.5;
		cvLine(frame, p0, p1, CV_RGB(0, 0, 255), 2);
	}
	
	// compute right-side left-side lanes
	int lc = 0, rc = 0;
	for(int g=0; g<right.size(); g++){
		if(fabs(right[g].angle) >= 8 && fabs(right[g].angle) <= 14)
			rc++;

	}
	for(int g=0; g<left.size(); g++){
		if(fabs(left[g].angle) >= 8 && fabs(left[g].angle) <= 14)
			lc++;
	}

	// determine the lane belong
	if(lc > LANE_COUNT_THRESHOLD && rc > LANE_COUNT_THRESHOLD){
		result[current] = MIDDLE;
	}
	else if(lc > LANE_COUNT_THRESHOLD){
		result[current] = RIGHT;
	}
	else if(rc > LANE_COUNT_THRESHOLD){
		result[current] = LEFT;
	}
	else{
		result[current] = NONE;
	}
	current = (current+1) % size;
	
	int temp[5] = {0};
	for(int i=0; i<size; i++){
		if(result[i] == LEFT)	temp[1]++;
		else if(result[i] == MIDDLE)	temp[2]++;
		else if(result[i] == RIGHT)		temp[3]++;
		else if(result[i] == NONE)	temp[4]++;
		//printf("%d ", result[i]);
	}
	//printf("\ntemp = %d %d %d %d\n", temp[1], temp[2], temp[3], temp[4]);
	int win = NONE, maxc = 0;
	for(int i=1; i<=4; i++){
		if(temp[i] > maxc){
			win = i;
			maxc = temp[i];
		}
	}

	/*
	if(win == LEFT)
		cvPutText(temp_frame, "left", cvPoint(850, 100), &font1, cvScalar(0, 255, 0));
	else if(win == MIDDLE)
		cvPutText(temp_frame, "middle", cvPoint(950, 100), &font1, cvScalar(0, 255, 0));
	else if(win == RIGHT)
		cvPutText(temp_frame, "right", cvPoint(1150, 100), &font1, cvScalar(0, 255, 0));
	else if(win == NONE)
		cvPutText(temp_frame, "none determine", cvPoint(870, 50), &font, cvScalar(0, 165, 255));
	*/

	return win;
}

int lane_detection(IplImage* input, int* result, int size, int &current){

	CvFont font;
	cvInitFont( &font, CV_FONT_VECTOR0, 0.25f, 0.25f);

	long current_frame = 0;
	int key_pressed = 0;

	CvSize frame_size = cvSize(input->width, input->height/3.5);
	IplImage *temp_frame = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
	IplImage *grey = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
	IplImage *edges = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);

	CvMemStorage* houghStorage = cvCreateMemStorage(0);
	CvMemStorage* haarStorage = cvCreateMemStorage(0);

	//cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, current_frame);

	// we're interested only in road below horizont - so crop top image portion off
	crop(input, temp_frame, cvRect(0,input->height - frame_size.height,frame_size.width,frame_size.height));
	cvCvtColor(temp_frame, grey, CV_BGR2GRAY); // convert to grayscale

	// Perform a Gaussian blur ( Convolving with 5 X 5 Gaussian) & detect edges
	cvSmooth(grey, grey, CV_GAUSSIAN, 5, 5);
	//cvSmooth(grey, grey, CV_MEDIAN, 5, 5);
	cvCanny(grey, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

	// do Hough transform to find lanes
	double rho = 1;
	double theta = CV_PI/180;
	CvSeq* lines = cvHoughLines2(edges, houghStorage, CV_HOUGH_PROBABILISTIC, rho, theta, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

	// find lanes
	int out = processLanes(lines, edges, temp_frame, input, result, size, current);
	
	return out;

	// show middle line
	//cvLine(temp_frame, cvPoint(CAR_MIDDLE,0), cvPoint(CAR_MIDDLE,frame_size.height), CV_RGB(255, 255, 0), 1);

	//cvShowImage("Grey", grey);
	//cvShowImage("Edges", edges);
	//cvShowImage("Color", temp_frame);

	//cvMoveWindow("Grey", 0, 0); 
	//cvMoveWindow("Edges", 0, 0);
	//cvMoveWindow("Color", 0, 0); 

	/*
	cvReleaseMemStorage(&haarStorage);
	cvReleaseMemStorage(&houghStorage);

	cvReleaseImage(&grey);
	cvReleaseImage(&edges);
	cvReleaseImage(&temp_frame);
	*/
}
