#ifndef REGIONUPDATOR_H
#define REGIONUPDATOR_H

#include "cv_header.h"

class RegionUpdator
{
public:
	RegionUpdator(int w, int h);
	bool setDiffThreshold(int th);
	bool setSkinThreshold(int hue_max, int hue_min, int gray_max, int gray_min);
	const IplImage* detect(IplImage* img);
	
protected:
	int index;

	int width;
	int height;

	int gray_th;
	int skin_hue_max;
	int skin_hue_min;
	int skin_gray_max;
	int skin_gray_min;

	IplImage* img_last_gray;
	IplImage* img_last_mask;
	IplImage* img_cur_gray;
	IplImage* img_cur_mask;

	IplImage* imgHSV; //HSV image
	IplImage* imgHue; //color
};

#endif
