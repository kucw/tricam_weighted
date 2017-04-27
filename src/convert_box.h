#ifndef CONVERT_BOX_H
#define CONVERT_BOX_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "image.h"

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
void get_demorgan_box_color(float* rgb, int obj_class, int CLS_NUM);

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

#endif

