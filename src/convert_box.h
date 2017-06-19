#ifndef CONVERT_BOX_H
#define CONVERT_BOX_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "image.h"

//======================================================
//		convert all box in others ROI to Left ROI
//======================================================
void convert_allrightbox_to_leftROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM);
void convert_alltopbox_to_rightROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM);

//==================================
//		convert box to other ROI
//==================================
// left <--> right
void convert_leftbox_to_rightROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM);
void convert_rightbox_to_leftROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM);

// left <--> up
void convert_leftbox_to_upROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM);
void convert_upbox_to_leftROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM);

// up <--> right
void convert_upbox_to_rightROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM);
void convert_rightbox_to_upROI(image det, float prob, box *boxes, image *labels, int i, int obj_class, int CLS_NUM);

//==================================
//		get demorgan box color
//==================================
void get_normal_box_color(float* rgb, int obj_class, int CLS_NUM);
void get_demorgan_box_color(float* rgb, int obj_class, int CLS_NUM);
void get_weighted_demorgan_box_color(float* rgb, int obj_class, int CLS_NUM);
void get_weighted_power_demorgan_box_color(float* rgb, int obj_class, int CLS_NUM);

//==================================
//		get box in other ROI
//==================================
// left
void get_leftbox_in_leftROI(image det, int *output, float prob, box *boxes, int i);
void get_rightbox_in_leftROI(image det, int *output, float prob, box *boxes, int i);
void get_upbox_in_leftROI(image det, int *output, float prob, box *boxes, int i);

// right
void get_leftbox_in_rightROI(image det, int *output, float prob, box *boxes, int i);
void get_rightbox_in_rightROI(image det, int *output, float prob, box *boxes, int i);
void get_upbox_in_rightROI(image det, int *output, float prob, box *boxes, int i);

// up
void get_leftbox_in_upROI(image det, int *output, float prob, box *boxes, int i);
void get_rightbox_in_upROI(image det, int *output, float prob, box *boxes, int i);
void get_upbox_in_upROI(image det, int *output, float prob, box *boxes, int i);

//==================================
//		get box union area
//==================================
float get_double_box_overlap(int lx, int rx, int ly, int ry, int *box1, int *box2);
float get_triple_box_overlap(int lx, int rx, int ly, int ry, int *box1, int *box2, int *box3);


#endif

