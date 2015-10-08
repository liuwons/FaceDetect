#ifndef FACEDETECT_H
#define FACEDETECT_H

#include <vector>

#include "MoveDetect.h"
#include "SkinDetect.h"

#include "Classifier.h"

using std::vector;

struct histogram
{
	int bin[256];
	int map[256];
	int pix_count;

	void clear()
	{
		memset(bin, 0, 256*sizeof(int));
		memset(map, 0, 256*sizeof(int));
		pix_count = 0;
	}

	void normalize()
	{
		float prob[256];
		prob[0] = bin[0];
		for (int i = 1; i < 256; i++)
		{
			prob[i] = bin[i] + prob[i-1];
			//cout << bin[i] << ",";
		}
		for (int i = 0; i < 256; i++)
		{
			prob[i] = prob[i] / pix_count;
		}

		for (int i = 0; i < 256; i++)
		{
			map[i] = prob[i] * 255;
			//cout << i << " map to " << map[i] << endl;
		}
	}
};

class FaceDetector
{
    private:
        string cascade_path;
        MaskCascadeClassifier cascade;

        MoveDetector* md;
        SkinDetector* sd;
        
        IplImage* imgMask;
        IplImage* imgGray;
        IplImage* imgBack;
		IplImage* imgCont; // for contour analyze
       
        int width, height;
        int fps;
        int index;

        int imgBufLen;
        IplImage** imgBuf;
        IntImage* ii;

		vector<Rect> last_detected;
		histogram hist;

    public:
        const static string DEFAULT_CASCADE_PATH;// = "haarcascade_frontalface_alt.xml"; 
        const static int MIN_SIZE;

        FaceDetector(int w, int h, int fp, int fb, const char* cp = 0);
        vector<Rect> detect(IplImage* img, IplImage* mask, CvRect ori);
        vector<Rect> detectAll(IplImage* img, CvRect* searched = 0);
        IplImage* getMask(IplImage* img);
		CvRect analyze(IplImage* mask, int th, IplImage* src);
};



#endif
