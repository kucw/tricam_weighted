#ifndef DEMORGAN_H
#define DEMORGAN_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "image.h"
#include "convert_box.h"

void Weighted_Demorgan_Power_right(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num, int frame_counter, int map[][450][5]);
void Weighted_Demorgan_Power_left(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num);
void Weighted_Demorgan_Power_up(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num);

void Weighted_Demorgan_right(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num);
void Weighted_Demorgan_left(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num);
void Weighted_Demorgan_up(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num);

void Demorgan_right(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num);
void Demorgan_left(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num);
void Demorgan_up(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num);

#endif

