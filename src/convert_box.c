#include "convert_box.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern IplImage* image_to_Ipl(image img, int w, int h, int depth, int c, int step);
extern image ipl_to_image(IplImage* src);

void convert_leftbox_to_rightROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM){
	int width = pow(prob, 1./2.)*10+1;
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color3(0,offset,CLS_NUM);
	float green = get_color3(1,offset,CLS_NUM);
	float blue = get_color3(2,offset,CLS_NUM);
	float rgb[3];
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;

	//resize the box
	left += 238;
	right += 238;
	top += 400;
	bot += 400;

	float a1 = 4.5;
	int b1 = -711;
	left = a1*left + b1;
	right = a1*right + b1;

	float a2 = 4.5;
	int b2 = -2354;
	top = a2*top + b2;
	bot = a2*bot + b2;
		
	//check if the box is in the M
	if (right < 1130 || left > 1578) return ;

	//clip the box
	if (left < 1130) left = 1130;
	if (right > 1578) right = 1578;
	if (top < 630) top = 630;
	if (bot > 1078) bot = 1078;

	if(labels) draw_label(det, top + width, left, labels[obj_class], rgb);
	draw_box_width(det, left, top, right, bot, width, red, green, blue);

	// draw probs
	char Text[30];
	sprintf(Text, "%.2f", prob);
	IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
	CvFont font2;
	CvPoint TextPos;
	TextPos.x = (left+right)/2; TextPos.y = top;
	cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
	cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
	image d = ipl_to_image(text);  
	memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
	free_image(d);
	cvReleaseImage(&text);
}

void convert_rightbox_to_leftROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM){
	int width = pow(prob, 1./2.)*10+1;
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color5(0,offset,CLS_NUM);
	float green = get_color5(1,offset,CLS_NUM);
	float blue = get_color5(2,offset,CLS_NUM);
	float rgb[3];
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;

	//resize the box
	left += 1130;
	right += 1130;
	top += 630;
	bot += 630;

	float a1 = 4.5;
	int b1 = -711;
	left = (left - b1)/a1;
	right = (right - b1)/a1;

	float a2 = 4.5;
	int b2 = -2354;
	top = (top - b2)/a2;
	bot = (bot - b2)/a2;
		
	//check if the box is in the M
	if (right < 238 || left > 676) return ;

	//clip the box
	if (left < 238) left = 238;
	if (right > 676) right = 676;
	if (top < 550) top = 550;
	if (bot > 998) bot = 998;

	if(labels) draw_label(det, top + width, left, labels[obj_class], rgb);
	draw_box_width(det, left, top, right, bot, width, red, green, blue);

	// draw probs
	char Text[30];
	sprintf(Text, "%.2f", prob);
	IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
	CvFont font2;
	CvPoint TextPos;
	TextPos.x = (left+right)/2; TextPos.y = top;
	cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
	cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
	image d = ipl_to_image(text);  
	memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
	free_image(d);
	cvReleaseImage(&text);
}


void convert_leftbox_to_upROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM){
	int width = pow(prob, 1./2.)*10+1;
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color3(0,offset,CLS_NUM);
	float green = get_color3(1,offset,CLS_NUM);
	float blue = get_color3(2,offset,CLS_NUM);
	float rgb[3];
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;

	//resize the box
	left += 238;
	right += 238;
	top += 400;
	bot += 400;
	
	float a1 = 2.2;
	int b1 = 20;
	left = a1*left + b1;
	right = a1*right + b1;

	float a2 = 2.2;
	int b2 = -1362;
	top = a2*top + b2;
	bot = a2*bot + b2;
		
	//check if the box is in the M
	if (right < 770 || left > 1218) return ;

	//clip the box
	if (left < 770) left = 770;
	if (right > 1218) right = 1218;
	if (top < 80) top = 80;
	if (bot > 528) bot = 528;

	if(labels) draw_label(det, top + width, left, labels[obj_class], rgb);
	draw_box_width(det, left, top, right, bot, width, red, green, blue);

	// draw probs
	char Text[30];
	sprintf(Text, "%.2f", prob);
	IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
	CvFont font2;
	CvPoint TextPos;
	TextPos.x = (left+right)/2; TextPos.y = top;
	cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
	cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
	image d = ipl_to_image(text);  
	memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
	free_image(d);
	cvReleaseImage(&text);
}

void convert_upbox_to_leftROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM){
	int width = pow(prob, 1./2.)*10+1;
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color4(0,offset,CLS_NUM);
	float green = get_color4(1,offset,CLS_NUM);
	float blue = get_color4(2,offset,CLS_NUM);
	float rgb[3];
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	
	//resize the box
	left += 770;
	right += 770;
	top += 80;
	bot += 80;

	float a1 = 2.2;
	int b1 = 20;
	left = (left - b1)/a1;
	right = (right - b1)/a1;

	float a2 = 2.2;
	int b2 = -1362;
	top = (top - b2)/a2;
	bot = (bot - b2)/a2;

	
	//check if the box is in the M
	if (right < 238 || left > 676) return ;

	//clip the box
	if (left < 238) left = 238;
	if (right > 676) right = 676;
	if (top < 550) top = 550;
	if (bot > 998) bot = 998;

	if(labels) draw_label(det, top + width, left, labels[obj_class], rgb);
	draw_box_width(det, left, top, right, bot, width, red, green, blue);

	// draw probs
	char Text[30];
	sprintf(Text, "%.2f", prob);
	IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
	CvFont font2;
	CvPoint TextPos;
	TextPos.x = (left+right)/2; TextPos.y = top;
	cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
	cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
	image d = ipl_to_image(text);  
	memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
	free_image(d);
	cvReleaseImage(&text);
}

void convert_upbox_to_rightROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM){
	int width = pow(prob, 1./2.)*10+1;
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color4(0,offset,CLS_NUM);
	float green = get_color4(1,offset,CLS_NUM);
	float blue = get_color4(2,offset,CLS_NUM);
	float rgb[3];
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;

	//resize the box
	left += 770;
	right += 778;
	top += 80;
	bot += 80;
	
	float a1 = 2;
	int b1 = -742;
	left = a1*left + b1;
	right = a1*right + b1;

	float a2 = 2;
	int b2 = 415;
	top = a2*top + b2;
	bot = a2*bot + b2;
	
	//check if the box is in the M
	if (right < 1130 || left > 1578) return ;

	//clip the box
	if (left < 1130) left = 1130;
	if (right > 1578) right = 1578;
	if (top < 630) top = 630;
	if (bot > 1078) bot = 1078;

	if(labels) draw_label(det, top + width, left, labels[obj_class], rgb);
	draw_box_width(det, left, top, right, bot, width, red, green, blue);

	// draw probs
	char Text[30];
	sprintf(Text, "%.2f", prob);
	IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
	CvFont font2;
	CvPoint TextPos;
	TextPos.x = (left+right)/2; TextPos.y = top;
	cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
	cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
	image d = ipl_to_image(text);  
	memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
	free_image(d);
	cvReleaseImage(&text);
}


void convert_rightbox_to_upROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM){
	int width = pow(prob, 1./2.)*10+1;
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color5(0,offset,CLS_NUM);
	float green = get_color5(1,offset,CLS_NUM);
	float blue = get_color5(2,offset,CLS_NUM);
	float rgb[3];
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;

	//resize the box
	left += 1130;
	right += 1130;
	top += 630;
	bot += 630;
	
	float a1 = 2;
	int b1 = -742;
	left = (left - b1)/a1;
	right = (right - b1)/a1;

	float a2 = 2;
	int b2 = 415;
	top = (top - b2)/a2;
	bot = (bot - b2)/a2;
		
	//check if the box is in the M
	if (right < 770 || left > 1218) return ;

	//clip the box
	if (left < 770) left = 770;
	if (right > 1218) right = 1218;
	if (top < 80) top = 80;
	if (bot > 528) bot = 528;

	if(labels) draw_label(det, top + width, left, labels[obj_class], rgb);
	draw_box_width(det, left, top, right, bot, width, red, green, blue);

	// draw probs
	char Text[30];
	sprintf(Text, "%.2f", prob);
	IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
	CvFont font2;
	CvPoint TextPos;
	TextPos.x = (left+right)/2; TextPos.y = top;
	cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
	cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
	image d = ipl_to_image(text);  
	memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
	free_image(d);
	cvReleaseImage(&text);
}


void get_normal_box_color(float* rgb, int obj_class, int CLS_NUM){
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color(0,offset,CLS_NUM);
	float green = get_color(1,offset,CLS_NUM);
	float blue = get_color(2,offset,CLS_NUM);
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
}

void get_demorgan_box_color(float* rgb, int obj_class, int CLS_NUM){
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color2(0,offset,CLS_NUM);
	float green = get_color2(1,offset,CLS_NUM);
	float blue = get_color2(2,offset,CLS_NUM);
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
}

void get_weighted_demorgan_box_color(float* rgb, int obj_class, int CLS_NUM){
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color3(0,offset,CLS_NUM);
	float green = get_color3(1,offset,CLS_NUM);
	float blue = get_color3(2,offset,CLS_NUM);
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
}

void get_weighted_power_demorgan_box_color(float* rgb, int obj_class, int CLS_NUM){
	int offset = obj_class*17 % CLS_NUM;
	float red = get_color5(0,offset,CLS_NUM);
	float green = get_color5(1,offset,CLS_NUM);
	float blue = get_color5(2,offset,CLS_NUM);
	rgb[0] = red;
	rgb[1] = green;
	rgb[2] = blue;
}

void get_leftbox_in_leftROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	left += 238;
	right += 238;
	top += 400;
	bot += 400;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}

void get_rightbox_in_leftROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	
	//resize the box
	left += 1130;
	right += 1130;
	top += 630;
	bot += 630;

	float a1 = 4.5;
	int b1 = -711;
	left = (left - b1)/a1;
	right = (right - b1)/a1;

	float a2 = 4.5;
	int b2 = -2354;
	top = (top - b2)/a2;
	bot = (bot - b2)/a2;
		
	//check if the box is in the M
	if (right < 238 || left > 676) return ;

	//clip the box
	if (left < 238) left = 238;
	if (right > 676) right = 676;
	if (top < 550) top = 550;
	if (bot > 998) bot = 998;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}

void get_upbox_in_leftROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	
	//resize the box
	left += 770;
	right += 770;
	top += 80;
	bot += 80;

	float a1 = 2.2;
	int b1 = 20;
	left = (left - b1)/a1;
	right = (right - b1)/a1;

	float a2 = 2.2;
	int b2 = -1362;
	top = (top - b2)/a2;
	bot = (bot - b2)/a2;

	
	//check if the box is in the M
	if (right < 238 || left > 676) return ;

	//clip the box
	if (left < 238) left = 238;
	if (right > 676) right = 676;
	if (top < 550) top = 550;
	if (bot > 998) bot = 998;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}

void get_leftbox_in_rightROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	
	//resize the box
	left += 238;
	right += 238;
	top += 400;
	bot += 400;

	float a1 = 4.5;
	int b1 = -711;
	left = a1*left + b1;
	right = a1*right + b1;

	float a2 = 4.5;
	int b2 = -2354;
	top = a2*top + b2;
	bot = a2*bot + b2;
		
	//check if the box is in the M
	if (right < 1130 || left > 1578) return ;

	//clip the box
	if (left < 1130) left = 1130;
	if (right > 1578) right = 1578;
	if (top < 630) top = 630;
	if (bot > 1078) bot = 1078;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}

void get_rightbox_in_rightROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	left += 1130;
	right += 1130;
	top += 630;
	bot += 630;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}

void get_upbox_in_rightROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	
	//resize the box
	left += 770;
	right += 778;
	top += 80;
	bot += 80;
	
	float a1 = 2;
	int b1 = -742;
	left = a1*left + b1;
	right = a1*right + b1;

	float a2 = 2;
	int b2 = 415;
	top = a2*top + b2;
	bot = a2*bot + b2;
	
	//check if the box is in the M
	if (right < 1130 || left > 1578) return ;

	//clip the box
	if (left < 1130) left = 1130;
	if (right > 1578) right = 1578;
	if (top < 630) top = 630;
	if (bot > 1078) bot = 1078;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}

void get_leftbox_in_upROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	
	//resize the box
	left += 238;
	right += 238;
	top += 400;
	bot += 400;
	
	float a1 = 2.2;
	int b1 = 20;
	left = a1*left + b1;
	right = a1*right + b1;

	float a2 = 2.2;
	int b2 = -1362;
	top = a2*top + b2;
	bot = a2*bot + b2;
		
	//check if the box is in the M
	if (right < 770 || left > 1218) return ;

	//clip the box
	if (left < 770) left = 770;
	if (right > 1218) right = 1218;
	if (top < 80) top = 80;
	if (bot > 528) bot = 528;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}

void get_rightbox_in_upROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	
	//resize the box
	left += 1130;
	right += 1130;
	top += 630;
	bot += 630;
	
	float a1 = 2;
	int b1 = -742;
	left = (left - b1)/a1;
	right = (right - b1)/a1;

	float a2 = 2;
	int b2 = 415;
	top = (top - b2)/a2;
	bot = (bot - b2)/a2;
		
	//check if the box is in the M
	if (right < 770 || left > 1218) return ;

	//clip the box
	if (left < 770) left = 770;
	if (right > 1218) right = 1218;
	if (top < 80) top = 80;
	if (bot > 528) bot = 528;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}

void get_upbox_in_upROI(image det, int *output, float prob, box *boxes, int i){
	int width = pow(prob, 1./2.)*10+1;
	box b = boxes[i];
	int left,right,top,bot;
	left  = (b.x-b.w/2.)*448;
	right = (b.x+b.w/2.)*448;
	top   = (b.y-b.h/2.)*448;
	bot   = (b.y+b.h/2.)*448;
	if(left < 0) left = 0;
	if(right > det.w-1) right = 448-1;
	if(top < 0) top = 0;
	if(bot > det.h-1) bot = 448-1;
	
	left += 770;
	right += 770;
	top += 80;
	bot += 80;
	
	output[0] = left;
	output[1] = right;
	output[2] = top;
	output[3] = bot;
	output[4] = width;
}



#endif
