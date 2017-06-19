#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "lane_detection.cpp"
#include "unistd.h"
#include "utils.h"
extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "convert_box.h"
#include "demorgan.h"
#include "image.h"
#include "thpool.h"
#include <sys/time.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <math.h>
#include <stdio.h>
}

/* Change class number here */
#define CLS_NUM 20
#define RUN_TIMES 1

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern "C" IplImage* image_to_Ipl(image img, int w, int h, int depth, int c, int step);
extern "C" image ipl_to_image(IplImage* src);
extern "C" void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
extern "C" void draw_yolo(image im, int num, float thresh, box *boxes, float **probs);

extern "C" char *voc_names[];
extern "C" image voc_labels[];
extern "C" void draw_text(image a, char Text[], CvPoint TextPos);
#define RESULT_SIZE 55
static float **probs_left;
static float **probs_right;
static float **probs_up;
static box *boxes_left;
static box *boxes_right;
static box *boxes_up;
static network net;
static network net2;
static image in   ;
static image in_s ;
static image in_left;
static image in_right;
static image in_up;
static image det  ;
static image det_s;
static image det_left;
static image det_right;
static image det_up;
static image disp ;
static cv::VideoCapture cap;
static cv::VideoWriter cap_out;
static float demo_thresh = 0.2;
static float demo_thresh_low = 0.02;
static int w, h, depth, c, step= 0;
float FPS = 0;

int result[RESULT_SIZE] = {0};
int current = 0;
#define LEFT 1
#define MIDDLE 2
#define RIGHT 3
#define SINGLE 4
typedef struct ObjDetArg{
	image ROI;
	int draw;
}ODA;
int output;
int *control = (int*)malloc(sizeof(int));
int *traffic_mode = (int*)malloc(sizeof(int));

char mode[20];
char lane[20];
char fpss[20];
cv::Mat frames[1900];
struct timeval start_time;	

int right_falsepositive_map[450][450][5];
int left_falsepositive_map[450][450][5];

Rect rightbox_in_left[100];
Rect upbox_in_left[100];
Rect leftbox_in_left[100];
//int Rtree[10][300];	//assume there are 300 leaf box, M is 3, so level is 6



//========== control parameter ==================

//int frame_counter = 200; //demo frame counter
//int frame_counter = 1200;
int frame_counter = 0;
#define DRAW_CONVERT
#define DRAW_LOW_THRESHOLD_DETECTION
//#define DEMORGAN_RIGHT
//#define WEIGHTED_DEMORGAN_RIGHT
//#define WEIGHTED_POWER_DEMORGAN_RIGHT
//#define DEMORGAN_LEFT
//#define WEIGHTED_DEMORGAN_LEFT
#define WEIGHTED_POWER_DEMORGAN_LEFT
#define FRAME_BY_FRAME
//#define BOTH_DEMORGAN
//#define FALSE_POSITIVE_REMOVAL

//===============================================


//cv::Mat *frames = (cv::Mat*)malloc(sizeof(cv::Mat)*1800);

void *fetch_in_thread(void *Elastic)
{
	//int elastic = *((int*)Elastic);
	struct timeval now;
	
	gettimeofday(&now, NULL);
	int msec = (now.tv_sec - start_time.tv_sec)*1000 + (now.tv_usec - start_time.tv_usec)/1000;
	

	//frame_counter = msec/30;
	frame_counter++;
	IplImage frame = frames[frame_counter];

	if(step == 0)
	{
		w = frame.width;
		h = frame.height;
		c = frame.nChannels;
		depth= frame.depth; 
		step = frame.widthStep;
	}   

	
	int a, b, c;
	for(a=0; a<448; a++){
		for(b=0; b<448; b++){
			right_falsepositive_map[a][b][frame_counter%5] = 0;
			left_falsepositive_map[a][b][frame_counter%5] = 0;
		}
	}
	

	//output = lane_detection(&frame, result, RESULT_SIZE, current);

	in = ipl_to_image(&frame);
	rgbgr_image(in);
	in_s = resize_image(in, net.w, net.h);
	//in_left = crop_image(in,238,550,448,448);
	in_left = crop_image(in,238,400,448,448);	//reduce the car region
	in_right = crop_image(in,1130,630,448,448);
	in_up = crop_image(in,770,80,448,448);
	
	output = MIDDLE;
	if(output == MIDDLE){
		*control = 3;
		strcpy(lane, "");
		draw_box(in,238,550,676,998,0,0,0);
		draw_box(in,1130,630,1578,1078,0,0,0);
		draw_box(in,770,80,1218,528,0,0,0);
	}
	
	return 0;
}


int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}


void *detect_in_thread_up(void *arg)
{
	ODA tmp = *((ODA*)arg);
	float nms = .4;
	detection_layer l = net.layers[net.n-1];
	//show_image(tmp.ROI,"123");
	float *X = tmp.ROI.data;
	float *predictions = network_predict(net, X);

	//free_image(tmp.ROI);
	convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh_low, probs_up, boxes_up, 0);
	if (nms > 0) do_nms(boxes_up, probs_up, l.side*l.side*l.n, l.classes, nms);
#ifdef DRAW_LOW_THRESHOLD_DETECTION
	draw_detections(det, l.side*l.side*l.n, demo_thresh_low, boxes_up, probs_up, voc_names, voc_labels, CLS_NUM,tmp.draw);
#else
	draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes_up, probs_up, voc_names, voc_labels, CLS_NUM,tmp.draw);
#endif

#ifdef DRAW_CONVERT
	int i;
	int num = l.side*l.side*l.n; 
	for(i = 0; i < num; ++i){
		int obj_class = max_index(probs_up[i], CLS_NUM);
		if(voc_names[obj_class] == "car"){
			float prob_up = probs_up[i][obj_class];
			if(prob_up > demo_thresh_low){
				convert_upbox_to_leftROI(det, prob_up, boxes_up, voc_labels, i, obj_class, CLS_NUM);
				convert_upbox_to_rightROI(det, prob_up, boxes_up, voc_labels, i, obj_class, CLS_NUM);
			}
		}
	}
#endif

	//print MODE
	printf("\033[2J");
	printf("\033[1;1H");
	printf("\nframe_counter: %d\n", frame_counter);

	//print FPS
	printf("\nFPS:%.0f\n",FPS);
	sprintf(fpss, "FPS: %.0f", FPS);
	printf("Object:\n\n");
	

	return 0;
}

void *detect_in_thread_left(void *arg)
{
	ODA tmp = *((ODA*)arg);
	float nms = .4;
	detection_layer l = net.layers[net.n-1];
	//show_image(tmp.ROI,"123");
	float *X = tmp.ROI.data;
	float *predictions = network_predict(net, X);

	//free_image(tmp.ROI);
	convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh_low, probs_left, boxes_left, 0);
	if (nms > 0) do_nms(boxes_left, probs_left, l.side*l.side*l.n, l.classes, nms);

#ifdef FALSE_POSITIVE_REMOVAL
	//Remove false positive
	int i;
	int num = l.side*l.side*l.n; 
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		if(voc_names[obj_class_left] == "car"){
			float prob_left = probs_left[i][obj_class_left];
			if(prob_left > demo_thresh){
				int box_left[5];
				int x, y, j, count = 0, box_left_area = 0;
				get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, i);
				box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);	
				
				for(y=box_left[2]; y<box_left[3]; y++){
					for(x=box_left[0]; x<box_left[1]; x++){
						int buffer_count = 0;
						for(j=0; j<5; j++){
							if(left_falsepositive_map[y-550][x-238][j] == 1)
								buffer_count++;
						}
						if(buffer_count >= 3)
							count++;
						left_falsepositive_map[y-550][x-238][frame_counter%5] = 1;
					}
				}
				
				float area = (float)count/(float)box_left_area;
				
				if(area > 0.6){	
					float rgb[3];
					get_normal_box_color(rgb, obj_class_left, CLS_NUM);
					if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
					draw_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_left);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_left[0]+box_left[1])/2-50; TextPos.y = box_left[2];
					cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
					cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
					image d = ipl_to_image(text);  
					memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
					free_image(d);
					cvReleaseImage(&text);										
				}
			}
		}
	}
#elif defined(DRAW_LOW_THRESHOLD_DETECTION)
	draw_detections(det, l.side*l.side*l.n, demo_thresh_low, boxes_left, probs_left, voc_names, voc_labels, CLS_NUM, tmp.draw);
#else
	draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes_left, probs_left, voc_names, voc_labels, CLS_NUM, tmp.draw);
#endif



#ifdef DRAW_CONVERT
	int i;
	int num = l.side*l.side*l.n; 
	for(i = 0; i < num; ++i){
		int obj_class = max_index(probs_left[i], CLS_NUM);
		if(voc_names[obj_class] == "car"){
			float prob_left = probs_left[i][obj_class];
			if(prob_left > demo_thresh_low){
				convert_leftbox_to_rightROI(det, prob_left, boxes_left, voc_labels, i, obj_class, CLS_NUM);
				convert_leftbox_to_upROI(det, prob_left, boxes_left, voc_labels, i, obj_class, CLS_NUM);
			}
		}
	}
#endif
	
	//print MODE
	printf("\033[2J");
	printf("\033[1;1H");
	printf("\nframe_counter: %d\n", frame_counter);

	//print FPS
	printf("\nFPS:%.0f\n",FPS);
	sprintf(fpss, "FPS: %.0f", FPS);
	printf("Object:\n\n");
	

	return 0;
}


void *detect_in_thread_right(void *arg)
{
	ODA tmp = *((ODA*)arg);
	float nms = .4;
	detection_layer l = net.layers[net.n-1];
	//show_image(tmp.ROI,"123");
	float *X = tmp.ROI.data;
	float *predictions = network_predict(net, X);

	//free_image(tmp.ROI);
	convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh_low, probs_right, boxes_right, 0);
	if (nms > 0) do_nms(boxes_right, probs_right, l.side*l.side*l.n, l.classes, nms);
	
	convert_allrightbox_to_leftROI(det, rightbox_in_left, probs_right, boxes_right, l.side*l.side*l.n, CLS_NUM);


#ifdef FALSE_POSITIVE_REMOVAL	
	//Remove false positive
	int i;
	int num = l.side*l.side*l.n; 
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		if(voc_names[obj_class_right] == "car"){
			float prob_right = probs_right[i][obj_class_right];
			if(prob_right > demo_thresh){
				int box_right[5];
				int x, y, j, count = 0, box_right_area = 0;
				get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, i);
				box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);	
				
				for(y=box_right[2]; y<box_right[3]; y++){
					for(x=box_right[0]; x<box_right[1]; x++){
						int buffer_count = 0;
						for(j=0; j<5; j++){
							if(right_falsepositive_map[y-630][x-1130][j] == 1)
								buffer_count++;
						}
						if(buffer_count >= 3)
							count++;
						right_falsepositive_map[y-630][x-1130][frame_counter%5] = 1;
					}
				}
				
				float area = (float)count/(float)box_right_area;
				
				if(area > 0.6){	
					float rgb[3];
					get_normal_box_color(rgb, obj_class_right, CLS_NUM);
					if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
					draw_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_right);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_right[0]+box_right[1])/2-50; TextPos.y = box_right[2];
					cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
					cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
					image d = ipl_to_image(text);  
					memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
					free_image(d);
					cvReleaseImage(&text);										
				}
			}
		}
	}
#elif defined(DRAW_LOW_THRESHOLD_DETECTION)
	draw_detections(det, l.side*l.side*l.n, demo_thresh_low, boxes_right, probs_right, voc_names, voc_labels, CLS_NUM, tmp.draw);
#else
	draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes_right, probs_right, voc_names, voc_labels, CLS_NUM, tmp.draw);
#endif


#ifdef DRAW_CONVERT
	int i;
	int num = l.side*l.side*l.n; 
	for(i = 0; i < num; ++i){
		int obj_class = max_index(probs_right[i], CLS_NUM);
		if(voc_names[obj_class] == "car"){
			float prob_right = probs_right[i][obj_class];
			if(prob_right > demo_thresh_low){
				convert_rightbox_to_leftROI(det, prob_right, boxes_right, voc_labels, i, obj_class, CLS_NUM);
				convert_rightbox_to_upROI(det, prob_right, boxes_right, voc_labels, i, obj_class, CLS_NUM);
			}
		}
	}
#endif

	//print MODE
	printf("\033[2J");
	printf("\033[1;1H");
	printf("\nframe_counter: %d\n", frame_counter);

	//print FPS
	printf("\nFPS:%.0f\n",FPS);
	sprintf(fpss, "FPS: %.0f", FPS);
	printf("Object:\n\n");
	

	return 0
		;
}

extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, char *videofile, char *version)
{
	//demo_thresh = thresh;
	printf("YOLO demo\n");
	net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);

	int total_frame;

	srand(2222222);
	if(cam_index != -1)
	{
		cv::VideoCapture cam(cam_index);
		cap = cam;
		//if(!cap.isOpened()) error("Couldn't connect to webcam.\n");
	}
	else 
	{
		printf("Video File name is: %s\n", videofile);
		cv::VideoCapture videoCap(videofile);
		cap = videoCap;
		//if(!cap.isOpened()) error("Couldn't read video file.\n");

		cv::Size S = cv::Size((int)videoCap.get(CV_CAP_PROP_FRAME_WIDTH), (int)videoCap.get(CV_CAP_PROP_FRAME_HEIGHT));
		//Preload all frames

		total_frame= (int)videoCap.get(CV_CAP_PROP_FRAME_COUNT);

		fprintf(stderr, "preLoad...\n");
		int frame_number = 0;
		for(;frame_number < total_frame;){
			cap >> frames[frame_number++];
			//if(cv::waitKey(0)>=0)break;
		}
		printf("Load OK.\n");
		
			
		//cv::VideoWriter outputVideo("out.avi", CV_FOURCC('D','I','V','X'), videoCap.get(CV_CAP_PROP_FPS), S, true);
		//if(!outputVideo.isOpened()) error("Couldn't write video file.\n");
		//cap_out = outputVideo;
	}
	
	int i, j, k;
	for(i=0; i<448; i++){
		for(j=0; j<448; j++){
			for(k=0; k<5; k++){
				right_falsepositive_map[i][j][k] = 0;
				left_falsepositive_map[i][j][k] = 0;
			}
		}
	}

	detection_layer l = net.layers[net.n-1];
	gettimeofday(&start_time, NULL);

	boxes_left = (box *)calloc(l.side*l.side*l.n, sizeof(box));
	boxes_right = (box *)calloc(l.side*l.side*l.n, sizeof(box));
	boxes_up = (box *)calloc(l.side*l.side*l.n, sizeof(box));
	probs_left = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
	probs_right = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
	probs_up = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
	for(j = 0; j < l.side*l.side*l.n; ++j){
		probs_left[j] = (float *)calloc(l.classes, sizeof(float *));
		probs_right[j] = (float *)calloc(l.classes, sizeof(float *));
		probs_up[j] = (float *)calloc(l.classes, sizeof(float *));
	}
	threadpool thpool_cpu = thpool_init(4);
	threadpool thpool_gpu = thpool_init(1);
	//pthread_t fetch_thread;
	//pthread_t detect_thread;
	ODA *arg = (ODA*)malloc(sizeof(ODA));
	fetch_in_thread(0);
	det = in;
	det_s = in_s;
	det_left = in_left;
	det_right = in_right;
	det_up = in_up;
	fetch_in_thread(arg);
	detect_in_thread_left(arg);
	disp = det;
	det = in;
	det_s = in_s;
	det_left = in_left;
	det_right = in_right;
	det_up = in_up;
	for (int k = 0; k < RUN_TIMES; ++k){
		//frame_counter = 250;
		do {
			struct timeval tval_before, tval_after, tval_result;	
			gettimeofday(&tval_before, NULL);
	
			if(*control == 3){
				thpool_add_work(thpool_cpu,fetch_in_thread,0);
				arg->ROI = det_left;
				arg->draw = 1;
				thpool_add_work(thpool_gpu,detect_in_thread_left,arg);
				thpool_wait(thpool_gpu);
				arg->ROI = det_right;
				arg->draw = 2;
				thpool_add_work(thpool_gpu,detect_in_thread_right,arg);
				thpool_wait(thpool_gpu);
				arg->ROI = det_up;
				arg->draw = 3;
				thpool_add_work(thpool_gpu,detect_in_thread_up,arg);
				thpool_wait(thpool_gpu);
				thpool_wait(thpool_cpu);

				

#ifdef WEIGHTED_DEMORGAN_LEFT
				Weighted_Demorgan_left(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
#endif
#ifdef WEIGHTED_POWER_DEMORGAN_LEFT
				Weighted_Demorgan_Power_left(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n, frame_counter, left_falsepositive_map);
#endif
#ifdef DEMORGAN_LEFT
				Demorgan_left(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
#endif


#ifdef WEIGHTED_DEMORGAN_RIGHT
				Weighted_Demorgan_right(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
#endif
#ifdef WEIGHTED_POWER_DEMORGAN_RIGHT
				Weighted_Demorgan_Power_right(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n, frame_counter, right_falsepositive_map);
#endif
#ifdef DEMORGAN_RIGHT
				Demorgan_right(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
#endif

#ifdef BOTH_DEMORGAN
				Weighted_Demorgan_Power_right(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
				Weighted_Demorgan_Power_left(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
				Weighted_Demorgan_Power_up(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
				Demorgan_right(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
				Demorgan_left(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
				Demorgan_up(det, demo_thresh, demo_thresh_low, probs_right, probs_left, probs_up, boxes_right, boxes_left, boxes_up, voc_names, voc_labels, CLS_NUM, l.side*l.side*l.n);
#endif
				
			}

			
    		//save_image(disp, "test");
			show_image_and_text(disp, "YOLO", mode, lane, fpss);
			free_image(disp);
#ifdef FRAME_BY_FRAME
			cvWaitKey(0);
#else
			cvWaitKey(1);
#endif
			thpool_wait(thpool_cpu);
			thpool_wait(thpool_gpu);
			disp  = det;
			free_image(det_s);
			free_image(det_left);
			free_image(det_right);
			free_image(det_up);
			det   = in;
			det_s = in_s;
			det_left = in_left;
			det_right = in_right;
			det_up = in_up;
			gettimeofday(&tval_after, NULL);
			timersub(&tval_after, &tval_before, &tval_result);
			float curr = 1000000.f/((long int)tval_result.tv_usec);
			FPS = .9*FPS + .1*curr;

		}while(frame_counter < total_frame - 30);
		struct timeval stop_time;
		gettimeofday(&stop_time, NULL);
		double msec = (stop_time.tv_sec - start_time.tv_sec)*1000 + (stop_time.tv_usec - start_time.tv_usec)/1000;
		printf("1 frame = %f ms\n", (double)msec/(double)frame_counter);

	}
}
#else
extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index){
	fprintf(stderr, "YOLO demo needs OpenCV for webcam images.\n");
}
#endif

