#include "convert_box.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern IplImage* image_to_Ipl(image img, int w, int h, int depth, int c, int step);
extern image ipl_to_image(IplImage* src);

void convert_allrightbox_to_leftROI(image det, Rect *output, float **probs, box *boxes, char **names, int demo_thresh_low, int num, int CLS_NUM){
	int i, k=0;	
	for(i = 0; i < num; i++){
		int obj_class = max_index(probs[i], CLS_NUM);
		if(names[obj_class] == "car"){
			float prob = probs[i][obj_class];
			if(prob > demo_thresh_low){
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
				if (right < 238 || left > 676) continue ;
				if (bot < 550 || top > 998) continue ;

				//clip the box
				if (left < 238) left = 238;
				if (right > 676) right = 676;
				if (top < 550) top = 550;
				if (bot > 998) bot = 998;

				output[k].left = left;
				output[k].right = right;
				output[k].top = top;
				output[k].bot = bot;
				output[k].cx = (left+right)/2;
				output[k].cy = (top+bot)/2;
				output[k].obj_class = obj_class;
				output[k].prob = prob;
				output[k].hilbert_value = get_hilbert_value(output[k].cx-238, output[k].cy-550);
				k++;
			}
		}
	}
}

void convert_allupbox_to_leftROI(image det, Rect *output, float **probs, box *boxes, char **names, int demo_thresh_low, int num, int CLS_NUM){
	int i, k=0;	
	for(i = 0; i < num; i++){
		int obj_class = max_index(probs[i], CLS_NUM);
		if(names[obj_class] == "car"){
			float prob = probs[i][obj_class];
			if(prob > demo_thresh_low){
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
				if (right < 238 || left > 676) continue ;
				if (bot < 550 || top > 998) continue ;

				//clip the box
				if (left < 238) left = 238;
				if (right > 676) right = 676;
				if (top < 550) top = 550;
				if (bot > 998) bot = 998;

				output[k].left = left;
				output[k].right = right;
				output[k].top = top;
				output[k].bot = bot;
				output[k].cx = (left+right)/2;
				output[k].cy = (top+bot)/2;
				output[k].obj_class = obj_class;
				output[k].prob = prob;
				output[k].hilbert_value = get_hilbert_value(output[k].cx-238, output[k].cy-550);
				k++;
			}
		}
	}
}

int get_hilbert_value(int cx, int cy){
	cy = 512 - cy;
	int n = 9;
	int x, y, s, d=0;
	for (s=n/2; s>0; s/=2) {
		x = (cx & s) > 0;
		y = (cy & s) > 0;
		d += s * s * ((3 * x) ^ y);
		rot(s, &cx, &cy, x, y);
	}
	return d;
}
void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }

        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}

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

int max(int a, int b){
	if (a>b)
		return a;
	else
		return b;
}
int min(int a, int b){
	if (a<b)
		return a;
	else
		return b;
}
float get_double_box_overlap(int roi_lx, int roi_rx, int roi_ly, int roi_ry, int *box1, int *box2){
	int lx, ly, rx, ry;
	int box_union_area, box1_area, box2_area;
	int intersaction;
	
	lx = max(max(box1[0], roi_lx), box2[0]);
	rx = min(min(box1[1], roi_rx), box2[1]);
	ly = max(max(box1[2], roi_ly), box2[2]);
	ry = min(min(box1[3], roi_ry), box2[3]);
	
	intersaction = (rx-lx)*(ry-ly);

	box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2]);
	box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2]);
	box_union_area = box1_area + box2_area - intersaction;

	return (float)intersaction/(float)box_union_area;
}

float get_triple_box_overlap(int roi_lx, int roi_rx, int roi_ly, int roi_ry, int *box1, int *box2, int *box3){
	int lx, ly, rx, ry;
	int box_union_area, box1_area, box2_area, box3_area;
	int inter1_area, inter2_area, inter3_area;
	int intersaction;

	lx = max(max(max(box1[0], roi_lx), box2[0]), box3[0]);
	rx = min(min(min(box1[1], roi_rx), box2[1]), box3[1]);
	ly = max(max(max(box1[2], roi_ly), box2[2]), box3[2]);
	ry = min(min(min(box1[3], roi_ry), box2[3]), box3[3]);

	intersaction = (rx-lx+1)*(ry-ly+1);

	lx = max(max(box1[0], roi_lx), box2[0]);
	rx = min(min(box1[1], roi_rx), box2[1]);
	ly = max(max(box1[2], roi_ly), box2[2]);
	ry = min(min(box1[3], roi_ry), box2[3]);
	inter1_area = (rx-lx)*(ry-ly);

	lx = max(max(box1[0], roi_lx), box3[0]);
	rx = min(min(box1[1], roi_rx), box3[1]);
	ly = max(max(box1[2], roi_ly), box3[2]);
	ry = min(min(box1[3], roi_ry), box3[3]);	
	inter2_area = (rx-lx)*(ry-ly);

	lx = max(max(box2[0], roi_lx), box3[0]);
	rx = min(min(box2[1], roi_rx), box3[1]);
	ly = max(max(box2[2], roi_ly), box3[2]);
	ry = min(min(box2[3], roi_ry), box3[3]);
	inter3_area = (rx-lx)*(ry-ly);

	box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2]);
	box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2]);
	box3_area = (box3[1] - box3[0]) * (box3[3] - box3[2]);
	box_union_area = box1_area + box2_area + box3_area - inter1_area - inter2_area - inter3_area + intersaction;

	return (float)intersaction/(float)box_union_area;
}

#endif
