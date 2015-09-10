#ifndef FACEDETECT_H
#define FACEDETECT_H

#include <vector>

#include "MoveDetect.h"
#include "SkinDetect.h"

using std::vector;

class FaceDetector
{
    private:
        string cascade_path;
        CascadeClassifier cascade;

        MoveDetector* md;
        SkinDetector* sd;
        
        IplImage* imgMask;
        IplImage* imgGray;
        IplImage* imgBack;
       
        int fps;
        int index;

        int imgBufLen;
        IplImage** imgBuf;

        int width, height;
    public:
        const static string DEFAULT_CASCADE_PATH;// = "haarcascade_frontalface_alt.xml"; 
        const static int MIN_SIZE;

        FaceDetector(int w, int h, int fp, int fb, const char* cp = 0);
        vector<Rect> detect(const IplImage* img, CvRect rect);
        vector<CvRect> getCandidateRect(const IplImage* img);
};

vector<CvRect> regionAnalyze(IplImage* imgMask, int min_size);

#endif
