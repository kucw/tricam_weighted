#include "demorgan.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern IplImage* image_to_Ipl(image img, int w, int h, int depth, int c, int step);
extern image ipl_to_image(IplImage* src);

//#define DRAW_OVERLAP_PROB
#define WEIGHT 10 //use 10~60 to define distance weight coefficient, 0 means normal

//***********************************************
//         Weighted De morgan law - Right
//***********************************************
void Weighted_Demorgan_right(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num){
	
	int i,j,k;
	float rgb[3] = {0.0};
	float overlap1_thresh = 0.5, overlap2_thresh = 0.25;

#if WEIGHT == 10
	float weight_left = 84.3/(84.3+65.2+15.2)*3;
	float weight_up = 65.2/(84.3+65.2+15.2)*3;
	float weight_right = 15.2/(84.3+65.2+15.2)*3;
#elif WEIGHT == 20
	float weight_left = 86.9/(86.9+85.2+83.2)*3;
	float weight_up = 85.2/(86.9+85.2+83.2)*3;
	float weight_right = 83.2/(86.9+85.2+83.2)*3;
#elif WEIGHT == 30
	float weight_left = 73.4/(73.4+86.9+85.4)*3;
	float weight_up = 86.9/(73.4+86.9+85.4)*3;
	float weight_right = 85.4/(73.4+86.9+85.4)*3;
#elif WEIGHT == 40
	float weight_left = 52.5/(52.5+86.7+84.6)*3;
	float weight_up = 86.7/(52.5+86.7+84.6)*3;
	float weight_right = 84.6/(52.5+86.7+84.6)*3;
#elif WEIGHT == 50
	float weight_left = 16.5/(16.5+70.6+84.8)*3;
	float weight_up = 70.6/(16.5+70.6+84.8)*3;
	float weight_right = 84.8/(16.5+70.6+84.8)*3;
#elif WEIGHT == 60
	float weight_left = 4.5/(4.5+42.3+84.3)*3;
	float weight_up = 42.3/(4.5+42.3+84.3)*3;
	float weight_right = 84.3/(4.5+42.3+84.3)*3;
#elif WEIGHT == 70
	float weight_left = 0/(0+13.1+65.7)*3;
	float weight_up = 13.1/(0+13.1+65.7)*3;
	float weight_right = 65.7/(0+13.1+65.7)*3;
#elif WEIGHT == 0
	float weight_left = 1;
	float weight_up = 1;
	float weight_right = 1;
#endif

	//enhance the prediction
	// for each box in right
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		float prob_right = probs_right[i][obj_class_right];
		get_weighted_demorgan_box_color(rgb, obj_class_right, CLS_NUM);
		if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low && prob_right < demo_thresh){
			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_up[5] = {0}, box_right[5] = {0};
					int x, y, box_right_area = 0, box_up_area = 0;
					int intersaction1 = 0;
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, j);
					get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, i);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
								&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_right_area + box_up_area - intersaction1);
		
#ifdef DRAW_OVERLAP_PROB
					// draw overlap1 percentage
					char Text[30];
					sprintf(Text, "overlap1: %.2f%s", overlap1*100,"%");
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
					TextPos.x = 1250; TextPos.y = 580;
					cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
					cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
					image d = ipl_to_image(text);  
					memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
					free_image(d);
					cvReleaseImage(&text);
#endif

					//if intersaction overlap1 > thresh, do Demorgan law to all box in left
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_left = max_index(probs_left[k], CLS_NUM);
							float prob_left = probs_left[k][obj_class_left];
							if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
								int box_left[5] = {0};
								int x, y, box_left_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, k);
								box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

								for (y = 630; y<=1078; y++){
									for (x = 1130; x<=1578 ; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
#ifdef DRAW_OVERLAP_PROB
								// draw overlap percentage
								char Text[30];
								sprintf(Text, "overlap2: %.2f%s", overlap2*100, "%");
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								TextPos.x = 1250; TextPos.y = 620;
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);
#endif				
								
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-weight_up*prob_up) * (1-weight_right*prob_right) * (1-weight_left*prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
										draw_weighted_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_right*prob_right) * (1-weight_up*prob_up);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
								draw_weighted_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}

	// for each box in right
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		float prob_right = probs_right[i][obj_class_right];
		get_weighted_demorgan_box_color(rgb, obj_class_right, CLS_NUM);
		if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low && prob_right < demo_thresh){
			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_right[5] = {0};
					int x, y, box_right_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, j);
					get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, i);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_right_area + box_left_area - intersaction1);
		
					//if intersaction overlap1 > overlap1_thresh, do Demorgan law to all box in up
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_up = max_index(probs_up[k], CLS_NUM);
							float prob_up = probs_up[k][obj_class_up];
							if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
								int box_up[5] = {0};
								int x, y, box_up_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, k);
								box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

								for (y = 630; y<=1078; y++){
									for (x = 1130; x<=1578 ; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-weight_up*prob_up) * (1-weight_right*prob_right) * (1-weight_left*prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
										draw_weighted_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_right*prob_right) * (1-weight_left*prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
								draw_weighted_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}
	
	
	// for each box in up
	for(i = 0; i < num; ++i){
		int obj_class_up = max_index(probs_up[i], CLS_NUM);
		float prob_up = probs_up[i][obj_class_up];
		get_weighted_demorgan_box_color(rgb, obj_class_up, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_right = max_index(probs_right[k], CLS_NUM);
							float prob_right = probs_right[k][obj_class_right];
							if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
								int box_right[5] = {0};
								int x, y, box_right_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, k);
								box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

								for (y = 630; y<=1078; y++){
									for (x = 1130; x<=1578; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
								}
							}
						}

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_up*prob_up) * (1-weight_left*prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
								draw_weighted_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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
			}


			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);

					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}
		

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_up
				if (prob_up > demo_thresh){
					int box_up[5] = {0};
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, i);
					if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
					draw_weighted_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_up);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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
	
    // for each box in left
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		float prob_left = probs_left[i][obj_class_left];
		get_weighted_demorgan_box_color(rgb, obj_class_left, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
					}
				}
			}

			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, j);
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_right_area - intersaction1);
	
					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_left
				if (prob_left > demo_thresh){
					int box_left[5] = {0};
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, i);
                    if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
                    draw_weighted_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_left);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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
}


//***********************************************
//         Weighted De morgan law - Left
//***********************************************
void Weighted_Demorgan_left(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num){
	
	int i,j,k;
	float rgb[3] = {0.0};
	float overlap1_thresh = 0.5, overlap2_thresh = 0.25;
	
#if WEIGHT  == 10
	float weight_left = 84.3/(84.3+65.2+15.2)*3;
	float weight_up = 65.2/(84.3+65.2+15.2)*3;
	float weight_right = 15.2/(84.3+65.2+15.2)*3;
#elif WEIGHT == 20
	float weight_left = 86.9/(86.9+85.2+83.2)*3;
	float weight_up = 85.2/(86.9+85.2+83.2)*3;
	float weight_right = 83.2/(86.9+85.2+83.2)*3;
#elif WEIGHT == 30
	float weight_left = 73.4/(73.4+86.9+85.4)*3;
	float weight_up = 86.9/(73.4+86.9+85.4)*3;
	float weight_right = 85.4/(73.4+86.9+85.4)*3;
#elif WEIGHT == 40
	float weight_left = 52.5/(52.5+86.7+84.6)*3;
	float weight_up = 86.7/(52.5+86.7+84.6)*3;
	float weight_right = 84.6/(52.5+86.7+84.6)*3;
#elif WEIGHT == 50
	float weight_left = 16.5/(16.5+70.6+84.8)*3;
	float weight_up = 70.6/(16.5+70.6+84.8)*3;
	float weight_right = 84.8/(16.5+70.6+84.8)*3;
#elif WEIGHT == 60
	float weight_left = 4.5/(4.5+42.3+84.3)*3;
	float weight_up = 42.3/(4.5+42.3+84.3)*3;
	float weight_right = 84.3/(4.5+42.3+84.3)*3;
#elif WEIGHT == 70
	float weight_left = 0/(0+13.1+65.7)*3;
	float weight_up = 13.1/(0+13.1+65.7)*3;
	float weight_right = 65.7/(0+13.1+65.7)*3;
#elif WEIGHT == 0
	float weight_left = 1;
	float weight_up = 1;
	float weight_right = 1;
#endif
	
	//enhance the prediction
	// for each box in left
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		float prob_left = probs_left[i][obj_class_left];
		get_weighted_demorgan_box_color(rgb, obj_class_left, CLS_NUM);
		if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low && prob_left < demo_thresh){
			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, j);
					get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_right_area - intersaction1);
		
#ifdef DRAW_OVERLAP_PROB
					// draw overlap1 percentage
					char Text[30];
					sprintf(Text, "overlap1: %.2f%s", overlap1*100,"%");
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
					TextPos.x = 358; TextPos.y = 500;
					cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
					cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
					image d = ipl_to_image(text);  
					memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
					free_image(d);
					cvReleaseImage(&text);
#endif

					//if intersaction overlap1 > thresh, do Demorgan law to all box in up
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_up = max_index(probs_up[k], CLS_NUM);
							float prob_up = probs_up[k][obj_class_up];
							if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
								int box_up[5] = {0};
								int x, y, box_up_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, k);
								box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

								for (y = 550; y<=998; y++){
									for (x = 238; x<=676; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
#ifdef DRAW_OVERLAP_PROB
								// draw overlap percentage
								char Text[30];
								sprintf(Text, "overlap2: %.2f%s", overlap2*100, "%");
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								TextPos.x = 358; TextPos.y = 540;
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);
#endif				
								
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-weight_up*prob_up) * (1-weight_right*prob_right) * (1-weight_left*prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
										draw_weighted_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_left*prob_left) * (1-weight_right*prob_right);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
								draw_weighted_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}

	// for each box in left
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		float prob_left = probs_left[i][obj_class_left];
		get_weighted_demorgan_box_color(rgb, obj_class_left, CLS_NUM);
		if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low && prob_left < demo_thresh){
			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_up[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_up_area = 0;
					int intersaction1 = 0;
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, j);
					get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_up_area - intersaction1);
		
					//if intersaction overlap1 > overlap1_thresh, do Demorgan law to all box in right
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_right = max_index(probs_right[k], CLS_NUM);
							float prob_right = probs_right[k][obj_class_right];
							if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
								int box_right[5] = {0};
								int x, y, box_right_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, k);
								box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

								for (y = 550; y<=998; y++){
									for (x = 238; x<=676; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-weight_up*prob_up) * (1-weight_right*prob_right) * (1-weight_left*prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
										draw_weighted_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_up*prob_up) * (1-weight_left*prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
								draw_weighted_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}
	
	
	// for each box in up
	for(i = 0; i < num; ++i){
		int obj_class_up = max_index(probs_up[i], CLS_NUM);
		float prob_up = probs_up[i][obj_class_up];
		get_weighted_demorgan_box_color(rgb, obj_class_up, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);
		
                    //if intersaction overlap1 > thresh, do Demorgan law to all box in left
					if(overlap1 > overlap1_thresh){
						flag--;
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_left = max_index(probs_left[k], CLS_NUM);
							float prob_left = probs_left[k][obj_class_left];
							if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
								int box_left[5] = {0};
								int x, y, box_left_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, k);
								box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

								for (y = 550; y<=998; y++){
									for (x = 238; x<=676; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
								}
							}
						}

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_up*prob_up) * (1-weight_right*prob_right);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
								draw_weighted_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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
			}


			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);

					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}
		

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_up
				if (prob_up > demo_thresh){
					int box_up[5] = {0};
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, i);
					if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
					draw_weighted_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_up);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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
	
    // for each box in right
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		float prob_right = probs_right[i][obj_class_right];
		get_weighted_demorgan_box_color(rgb, obj_class_right, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
					}
				}
			}

			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, j);
					get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_right_area - intersaction1);
	
					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_right
				if (prob_right > demo_thresh){
					int box_right[5] = {0};
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, i);
                    if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
                    draw_weighted_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_right);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
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
}


//***********************************************
//         Weighted De morgan law - Up
//***********************************************
void Weighted_Demorgan_up(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num){
	
	int i,j,k;
	float rgb[3] = {0.0};
	float overlap1_thresh = 0.5, overlap2_thresh = 0.25;
	
#if WEIGHT  == 10
	float weight_left = 84.3/(84.3+65.2+15.2)*3;
	float weight_up = 65.2/(84.3+65.2+15.2)*3;
	float weight_right = 15.2/(84.3+65.2+15.2)*3;
#elif WEIGHT == 20
	float weight_left = 86.9/(86.9+85.2+83.2)*3;
	float weight_up = 85.2/(86.9+85.2+83.2)*3;
	float weight_right = 83.2/(86.9+85.2+83.2)*3;
#elif WEIGHT == 30
	float weight_left = 73.4/(73.4+86.9+85.4)*3;
	float weight_up = 86.9/(73.4+86.9+85.4)*3;
	float weight_right = 85.4/(73.4+86.9+85.4)*3;
#elif WEIGHT == 40
	float weight_left = 52.5/(52.5+86.7+84.6)*3;
	float weight_up = 86.7/(52.5+86.7+84.6)*3;
	float weight_right = 84.6/(52.5+86.7+84.6)*3;
#elif WEIGHT == 50
	float weight_left = 16.5/(16.5+70.6+84.8)*3;
	float weight_up = 70.6/(16.5+70.6+84.8)*3;
	float weight_right = 84.8/(16.5+70.6+84.8)*3;
#elif WEIGHT == 60
	float weight_left = 4.5/(4.5+42.3+84.3)*3;
	float weight_up = 42.3/(4.5+42.3+84.3)*3;
	float weight_right = 84.3/(4.5+42.3+84.3)*3;
#elif WEIGHT == 70
	float weight_left = 0/(0+13.1+65.7)*3;
	float weight_up = 13.1/(0+13.1+65.7)*3;
	float weight_right = 65.7/(0+13.1+65.7)*3;
#elif WEIGHT == 0
	float weight_left = 1;
	float weight_up = 1;
	float weight_right = 1;
#endif
	
	//enhance the prediction
	// for each box in up
	for(i = 0; i < num; ++i){
		int obj_class_up = max_index(probs_up[i], CLS_NUM);
		float prob_up = probs_up[i][obj_class_up];
		get_weighted_demorgan_box_color(rgb, obj_class_up, CLS_NUM);
		if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low && prob_up < demo_thresh){
			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_upROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);
		
#ifdef DRAW_OVERLAP_PROB
					// draw overlap1 percentage
					char Text[30];
					sprintf(Text, "overlap1: %.2f%s", overlap1*100,"%");
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
					TextPos.x = 890; TextPos.y = 30;
					cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
					cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
					image d = ipl_to_image(text);  
					memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
					free_image(d);
					cvReleaseImage(&text);
#endif

					//if intersaction overlap1 > thresh, do Demorgan law to all box in right
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_right = max_index(probs_right[k], CLS_NUM);
							float prob_right = probs_right[k][obj_class_right];
							if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
								int box_right[5] = {0};
								int x, y, box_right_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, k);
								box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

								for (y = 80; y<=528; y++){
									for (x = 780; x<=1218; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
#ifdef DRAW_OVERLAP_PROB
								// draw overlap percentage
								char Text[30];
								sprintf(Text, "overlap2: %.2f%s", overlap2*100, "%");
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								TextPos.x = 890; TextPos.y = 70;
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);
#endif				
								
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-weight_up*prob_up) * (1-weight_right*prob_right) * (1-weight_left*prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
										draw_weighted_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_up*prob_up) * (1-weight_left*prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
								draw_weighted_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}

	// for each box in up
	for(i = 0; i < num; ++i){
		int obj_class_up = max_index(probs_up[i], CLS_NUM);
		float prob_up = probs_up[i][obj_class_up];
		get_weighted_demorgan_box_color(rgb, obj_class_up, CLS_NUM);
		if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low && prob_up < demo_thresh){
			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_upROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);
		
					//if intersaction overlap1 > overlap1_thresh, do Demorgan law to all box in left
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_left = max_index(probs_left[k], CLS_NUM);
							float prob_left = probs_left[k][obj_class_left];
							if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
								int box_left[5] = {0};
								int x, y, box_left_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, k);
								box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

								for (y = 80; y<=528; y++){
									for (x = 780; x<=1218; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-weight_up*prob_up) * (1-weight_right*prob_right) * (1-weight_left*prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
										draw_weighted_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_right*prob_right) * (1-weight_up*prob_up);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
								draw_weighted_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}
	
	
	// for each box in left
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		float prob_left = probs_left[i][obj_class_left];
		get_weighted_demorgan_box_color(rgb, obj_class_left, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, j);
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_right_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_up = max_index(probs_up[k], CLS_NUM);
							float prob_up = probs_up[k][obj_class_up];
							if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
								int box_up[5] = {0};
								int x, y, box_up_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_upbox_in_upROI(det, box_up, prob_up, boxes_up, k);
								box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

								for (y = 80; y<=528; y++){
									for (x = 780; x<=1218; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
								}
							}
						}

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-weight_right*prob_right) * (1-weight_left*prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
								draw_weighted_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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
			}


			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_upROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);

					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}
		

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_left
				if (prob_left > demo_thresh){
					int box_left[5] = {0};
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, i);
					if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
					draw_weighted_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_left);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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
	
    // for each box in right
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		float prob_right = probs_right[i][obj_class_right];
		get_weighted_demorgan_box_color(rgb, obj_class_right, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_right[5] = {0};
					int x, y, box_right_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, j);
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, i);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_right_area + box_left_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
					}
				}
			}

			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_upROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);
	
					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_right
				if (prob_right > demo_thresh){
					int box_right[5] = {0};
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, i);
                    if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
                    draw_weighted_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_right);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
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
}




//==========================================
//			De morgan law - Right
//==========================================
void Demorgan_right(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num){
	
	int i,j,k;
	float rgb[3] = {0.0};
	float overlap1_thresh = 0.5, overlap2_thresh = 0.25;
	
	
	//enhance the prediction
	// for each box in right
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		float prob_right = probs_right[i][obj_class_right];
		get_demorgan_box_color(rgb, obj_class_right, CLS_NUM);
		if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low && prob_right < demo_thresh){
			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_up[5] = {0}, box_right[5] = {0};
					int x, y, box_right_area = 0, box_up_area = 0;
					int intersaction1 = 0;
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, j);
					get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, i);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
								&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_right_area + box_up_area - intersaction1);
		
#ifdef DRAW_OVERLAP_PROB
					// draw overlap1 percentage
					char Text[30];
					sprintf(Text, "overlap1: %.2f%s", overlap1*100,"%");
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
					TextPos.x = 1250; TextPos.y = 580;
					cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
					cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
					image d = ipl_to_image(text);  
					memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
					free_image(d);
					cvReleaseImage(&text);
#endif

					//if intersaction overlap1 > thresh, do Demorgan law to all box in left
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_left = max_index(probs_left[k], CLS_NUM);
							float prob_left = probs_left[k][obj_class_left];
							if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
								int box_left[5] = {0};
								int x, y, box_left_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, k);
								box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

								for (y = 630; y<=1078; y++){
									for (x = 1130; x<=1578 ; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
#ifdef DRAW_OVERLAP_PROB
								// draw overlap percentage
								char Text[30];
								sprintf(Text, "overlap2: %.2f%s", overlap2*100, "%");
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								TextPos.x = 1250; TextPos.y = 620;
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);
#endif				
								
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-prob_up) * (1-prob_right) * (1-prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
										draw_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_right) * (1-prob_up);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
								draw_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}

	// for each box in right
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		float prob_right = probs_right[i][obj_class_right];
		get_demorgan_box_color(rgb, obj_class_right, CLS_NUM);
		if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low && prob_right < demo_thresh){
			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_right[5] = {0};
					int x, y, box_right_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, j);
					get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, i);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_right_area + box_left_area - intersaction1);
		
					//if intersaction overlap1 > overlap1_thresh, do Demorgan law to all box in up
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_up = max_index(probs_up[k], CLS_NUM);
							float prob_up = probs_up[k][obj_class_up];
							if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
								int box_up[5] = {0};
								int x, y, box_up_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, k);
								box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

								for (y = 630; y<=1078; y++){
									for (x = 1130; x<=1578 ; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-prob_up) * (1-prob_right) * (1-prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
										draw_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_right) * (1-prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
								draw_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}
	
	
	// for each box in up
	for(i = 0; i < num; ++i){
		int obj_class_up = max_index(probs_up[i], CLS_NUM);
		float prob_up = probs_up[i][obj_class_up];
		get_demorgan_box_color(rgb, obj_class_up, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_right = max_index(probs_right[k], CLS_NUM);
							float prob_right = probs_right[k][obj_class_right];
							if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
								int box_right[5] = {0};
								int x, y, box_right_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, k);
								box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

								for (y = 630; y<=1078; y++){
									for (x = 1130; x<=1578; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
								}
							}
						}

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_up) * (1-prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
								draw_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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
			}


			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);

					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}
		

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_up
				if (prob_up > demo_thresh){
					int box_up[5] = {0};
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, i);
					if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
					draw_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_up);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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
	
    // for each box in left
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		float prob_left = probs_left[i][obj_class_left];
		get_demorgan_box_color(rgb, obj_class_left, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_rightROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
					}
				}
			}

			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_rightROI(det, box_right, prob_right, boxes_right, j);
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 630; y<=1078; y++){
						for (x = 1130; x<=1578; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_right_area - intersaction1);
	
					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_left
				if (prob_left > demo_thresh){
					int box_left[5] = {0};
					get_leftbox_in_rightROI(det, box_left, prob_left, boxes_left, i);
                    if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
                    draw_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_left);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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
}

//==========================================
//			De morgan law - Left
//==========================================
void Demorgan_left(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num){
	
	int i,j,k;
	float rgb[3] = {0.0};
	float overlap1_thresh = 0.5, overlap2_thresh = 0.25;
	
	
	//enhance the prediction
	// for each box in left
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		float prob_left = probs_left[i][obj_class_left];
		get_demorgan_box_color(rgb, obj_class_left, CLS_NUM);
		if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low && prob_left < demo_thresh){
			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, j);
					get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_right_area - intersaction1);
		
#ifdef DRAW_OVERLAP_PROB
					// draw overlap1 percentage
					char Text[30];
					sprintf(Text, "overlap1: %.2f%s", overlap1*100,"%");
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
					TextPos.x = 358; TextPos.y = 500;
					cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
					cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
					image d = ipl_to_image(text);  
					memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
					free_image(d);
					cvReleaseImage(&text);
#endif

					//if intersaction overlap1 > thresh, do Demorgan law to all box in up
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_up = max_index(probs_up[k], CLS_NUM);
							float prob_up = probs_up[k][obj_class_up];
							if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
								int box_up[5] = {0};
								int x, y, box_up_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, k);
								box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

								for (y = 550; y<=998; y++){
									for (x = 238; x<=676; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
#ifdef DRAW_OVERLAP_PROB
								// draw overlap percentage
								char Text[30];
								sprintf(Text, "overlap2: %.2f%s", overlap2*100, "%");
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								TextPos.x = 358; TextPos.y = 540;
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);
#endif				
								
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-prob_up) * (1-prob_right) * (1-prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
										draw_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_left) * (1-prob_right);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
								draw_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}

	// for each box in left
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		float prob_left = probs_left[i][obj_class_left];
		get_demorgan_box_color(rgb, obj_class_left, CLS_NUM);
		if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low && prob_left < demo_thresh){
			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_up[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_up_area = 0;
					int intersaction1 = 0;
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, j);
					get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_up_area - intersaction1);
		
					//if intersaction overlap1 > overlap1_thresh, do Demorgan law to all box in right
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_right = max_index(probs_right[k], CLS_NUM);
							float prob_right = probs_right[k][obj_class_right];
							if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
								int box_right[5] = {0};
								int x, y, box_right_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, k);
								box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

								for (y = 550; y<=998; y++){
									for (x = 238; x<=676; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-prob_up) * (1-prob_right) * (1-prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
										draw_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_up) * (1-prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
								draw_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}
	
	
	// for each box in up
	for(i = 0; i < num; ++i){
		int obj_class_up = max_index(probs_up[i], CLS_NUM);
		float prob_up = probs_up[i][obj_class_up];
		get_demorgan_box_color(rgb, obj_class_up, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);
		
                    //if intersaction overlap1 > thresh, do Demorgan law to all box in left
					if(overlap1 > overlap1_thresh){
						flag--;
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_left = max_index(probs_left[k], CLS_NUM);
							float prob_left = probs_left[k][obj_class_left];
							if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
								int box_left[5] = {0};
								int x, y, box_left_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, k);
								box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

								for (y = 550; y<=998; y++){
									for (x = 238; x<=676; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
								}
							}
						}

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_up) * (1-prob_right);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
								draw_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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
			}


			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);

					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}
		

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_up
				if (prob_up > demo_thresh){
					int box_up[5] = {0};
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, i);
					if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
					draw_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_up);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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
	
    // for each box in right
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		float prob_right = probs_right[i][obj_class_right];
		get_demorgan_box_color(rgb, obj_class_right, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_leftROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
					}
				}
			}

			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, j);
					get_leftbox_in_leftROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 550; y<=998; y++){
						for (x = 238; x<=676; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_right_area - intersaction1);
	
					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_right
				if (prob_right > demo_thresh){
					int box_right[5] = {0};
					get_rightbox_in_leftROI(det, box_right, prob_right, boxes_right, i);
                    if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
                    draw_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_right);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
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
}


//==========================================
//			De morgan law - Up
//==========================================
void Demorgan_up(image det, float demo_thresh, float demo_thresh_low, float **probs_right, float **probs_left, float **probs_up, box *boxes_right, box *boxes_left, box *boxes_up, char**voc_names, image *voc_labels, int CLS_NUM, int num){
	
	int i,j,k;
	float rgb[3] = {0.0};
	float overlap1_thresh = 0.5, overlap2_thresh = 0.25;
	
	
	//enhance the prediction
	// for each box in up
	for(i = 0; i < num; ++i){
		int obj_class_up = max_index(probs_up[i], CLS_NUM);
		float prob_up = probs_up[i][obj_class_up];
		get_demorgan_box_color(rgb, obj_class_up, CLS_NUM);
		if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low && prob_up < demo_thresh){
			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_upROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);
		
#ifdef DRAW_OVERLAP_PROB
					// draw overlap1 percentage
					char Text[30];
					sprintf(Text, "overlap1: %.2f%s", overlap1*100,"%");
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
					TextPos.x = 890; TextPos.y = 30;
					cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
					cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
					image d = ipl_to_image(text);  
					memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
					free_image(d);
					cvReleaseImage(&text);
#endif

					//if intersaction overlap1 > thresh, do Demorgan law to all box in right
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_right = max_index(probs_right[k], CLS_NUM);
							float prob_right = probs_right[k][obj_class_right];
							if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
								int box_right[5] = {0};
								int x, y, box_right_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, k);
								box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

								for (y = 80; y<=528; y++){
									for (x = 780; x<=1218; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
#ifdef DRAW_OVERLAP_PROB
								// draw overlap percentage
								char Text[30];
								sprintf(Text, "overlap2: %.2f%s", overlap2*100, "%");
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								//TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
								TextPos.x = 890; TextPos.y = 70;
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0,255,0));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);
#endif				
								
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-prob_up) * (1-prob_right) * (1-prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
										draw_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_up) * (1-prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
								draw_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}

	// for each box in up
	for(i = 0; i < num; ++i){
		int obj_class_up = max_index(probs_up[i], CLS_NUM);
		float prob_up = probs_up[i][obj_class_up];
		get_demorgan_box_color(rgb, obj_class_up, CLS_NUM);
		if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low && prob_up < demo_thresh){
			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_upROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);
		
					//if intersaction overlap1 > overlap1_thresh, do Demorgan law to all box in left
					if(overlap1 > overlap1_thresh){
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_left = max_index(probs_left[k], CLS_NUM);
							float prob_left = probs_left[k][obj_class_left];
							if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
								int box_left[5] = {0};
								int x, y, box_left_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, k);
								box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

								for (y = 80; y<=528; y++){
									for (x = 780; x<=1218; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
									float demorgan;
									demorgan = 1 - (1-prob_up) * (1-prob_right) * (1-prob_left);

									//if demorgan law's prob > 0.2, draw box
									if(demorgan > 0.2){
										if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
										draw_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
										char Text[30];
										sprintf(Text, "%.2f", demorgan);
										IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
										CvFont font2;
										CvPoint TextPos;
										TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
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

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_right) * (1-prob_up);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_up[2] + box_up[4], box_up[0], voc_labels[obj_class_up], rgb);
								draw_box_width(det, box_up[0], box_up[2], box_up[1], box_up[3], box_up[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_up[0]+box_up[1])/2; TextPos.y = box_up[2];
								cvInitFont(&font2 , CV_FONT_HERSHEY_SIMPLEX , 1 , 1 , 1 , 3 , CV_AA);
								cvPutText(text , Text , TextPos , &font2 , CV_RGB(0, 133, 255));	
								image d = ipl_to_image(text);  
								memcpy(det.data,d.data,det.h*det.w*det.c*sizeof(float));
								free_image(d);
								cvReleaseImage(&text);										
							}
						}
					}

					// if the overlap1 < overlap1_thresh, continue
				}
			}
		}
	}
	
	
	// for each box in left
	for(i = 0; i < num; ++i){
		int obj_class_left = max_index(probs_left[i], CLS_NUM);
		float prob_left = probs_left[i][obj_class_left];
		get_demorgan_box_color(rgb, obj_class_left, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
			// do De morgan law to all box in right
			for (j = 0; j < num; j++){
				int obj_class_right = max_index(probs_right[j], CLS_NUM);
				float prob_right = probs_right[j][obj_class_right];
				if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_left[5] = {0};
					int x, y, box_left_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, j);
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, i);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_left_area + box_right_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
						int get_overlap2_flag = 0;
						for (k = 0; k < num; k++){
							int obj_class_up = max_index(probs_up[k], CLS_NUM);
							float prob_up = probs_up[k][obj_class_up];
							if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
								int box_up[5] = {0};
								int x, y, box_up_area = 0;
								int inter_right_up = 0, inter_left_up = 0, inter_left_right = 0, intersaction2 = 0;
								get_upbox_in_upROI(det, box_up, prob_up, boxes_up, k);
								box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);

								for (y = 80; y<=528; y++){
									for (x = 780; x<=1218; x++){
										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y )
											intersaction2++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_right_up++;

										if ( box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y
											&& box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y)
											inter_left_up++;

										if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
											&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y)
											inter_left_right++;
									}
								}

								float overlap2 = (float)intersaction2/(float)(box_right_area + box_up_area + box_left_area - inter_right_up - inter_left_up - inter_left_right + intersaction2);
						
								if(overlap2 > overlap2_thresh){
									get_overlap2_flag = 1;
								}
							}
						}

						// if all overlap1 > overlap1_thesh but overlap2 < overlap2_thresh
						if (get_overlap2_flag != 1){
							float demorgan;
							demorgan = 1 - (1-prob_right) * (1-prob_left);

							//if demorgan law's prob > 0.2, draw box
							if(demorgan > 0.2){
								if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
								draw_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
								char Text[30];
								sprintf(Text, "%.2f", demorgan);
								IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
								CvFont font2;
								CvPoint TextPos;
								TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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
			}


			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, j);
					get_upbox_in_upROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_left_area - intersaction1);

					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}
		

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_left
				if (prob_left > demo_thresh){
					int box_left[5] = {0};
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, i);
					if(voc_labels) draw_label(det, box_left[2] + box_left[4], box_left[0], voc_labels[obj_class_left], rgb);
					draw_box_width(det, box_left[0], box_left[2], box_left[1], box_left[3], box_left[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_left);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_left[0]+box_left[1])/2; TextPos.y = box_left[2];
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
	
    // for each box in right
	for(i = 0; i < num; ++i){
		int obj_class_right = max_index(probs_right[i], CLS_NUM);
		float prob_right = probs_right[i][obj_class_right];
		get_demorgan_box_color(rgb, obj_class_right, CLS_NUM);
		int flag = 2;
		if(voc_names[obj_class_right] == "car" && prob_right > demo_thresh_low){
			// do De morgan law to all box in left
			for (j = 0; j < num; j++){
				int obj_class_left = max_index(probs_left[j], CLS_NUM);
				float prob_left = probs_left[j][obj_class_left];
				if(voc_names[obj_class_left] == "car" && prob_left > demo_thresh_low){
					//check intersaction area
					int box_left[5] = {0}, box_right[5] = {0};
					int x, y, box_right_area = 0, box_left_area = 0;
					int intersaction1 = 0;
					get_leftbox_in_upROI(det, box_left, prob_left, boxes_left, j);
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, i);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);
					box_left_area = (box_left[1] - box_left[0]) * (box_left[3] - box_left[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_left[0] <= x && box_left[1] >= x && box_left[2]<= y && box_left[3] >= y
								&& box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_right_area + box_left_area - intersaction1);
		
					if(overlap1 > overlap1_thresh){
						flag--;
					}
				}
			}

			// do De morgan law to all box in up
			for (j = 0; j < num; j++){
				int obj_class_up = max_index(probs_up[j], CLS_NUM);
				float prob_up = probs_up[j][obj_class_up];
				if(voc_names[obj_class_up] == "car" && prob_up > demo_thresh_low){
					//check intersaction area
					int box_right[5] = {0}, box_up[5] = {0};
					int x, y, box_up_area = 0, box_right_area = 0;
					int intersaction1 = 0;
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, j);
					get_upbox_in_upROI(det, box_up, prob_up, boxes_up, i);
					box_up_area = (box_up[1] - box_up[0]) * (box_up[3] - box_up[2]);
					box_right_area = (box_right[1] - box_right[0]) * (box_right[3] - box_right[2]);

					for (y = 80; y<=528; y++){
						for (x = 780; x<=1218; x++){
							if ( box_right[0] <= x && box_right[1] >= x && box_right[2]<= y && box_right[3] >= y
								&& box_up[0] <= x && box_up[1] >= x && box_up[2]<= y && box_up[3] >= y )
								intersaction1++;
						}
					}

					float overlap1 = (float)intersaction1/(float)(box_up_area + box_right_area - intersaction1);
	
					if(overlap1 > overlap1_thresh){
						flag--;	
					}
				}
			}

			if (flag == 2){
				//if the others overlap1 < overlap1_thresh, check prob_right
				if (prob_right > demo_thresh){
					int box_right[5] = {0};
					get_rightbox_in_upROI(det, box_right, prob_right, boxes_right, i);
                    if(voc_labels) draw_label(det, box_right[2] + box_right[4], box_right[0], voc_labels[obj_class_right], rgb);
                    draw_box_width(det, box_right[0], box_right[2], box_right[1], box_right[3], box_right[4], rgb[0], rgb[1], rgb[2]);
					char Text[30];
					sprintf(Text, "%.2f", prob_right);
					IplImage *text = image_to_Ipl(det,det.w,det.h,IPL_DEPTH_8U,det.c,det.w*det.c);			
					CvFont font2;
					CvPoint TextPos;
					TextPos.x = (box_right[0]+box_right[1])/2; TextPos.y = box_right[2];
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
}


