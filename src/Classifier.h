#include "cv_header.h"
#include <opencv2/objdetect/objdetect.hpp>

#ifndef CLASSIFIER_H
#define CLASSIFIER_h

#include <iostream>
using namespace std;


#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

#define CV_TILTED_PTRS( p0, p1, p2, p3, tilted, rect, step )                        \
    /* (x, y) */                                                                    \
    (p0) = tilted + (rect).x + (step) * (rect).y,                                   \
    /* (x - h, y + h) */                                                            \
    (p1) = tilted + (rect).x - (rect).height + (step) * ((rect).y + (rect).height), \
    /* (x + w, y + w) */                                                            \
    (p2) = tilted + (rect).x + (rect).width + (step) * ((rect).y + (rect).width),   \
    /* (x + w - h, y + w + h) */                                                    \
    (p3) = tilted + (rect).x + (rect).width - (rect).height                         \
           + (step) * ((rect).y + (rect).width + (rect).height)

#define CALC_SUM_(p0, p1, p2, p3, offset) \
    ((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CALC_SUM(rect,offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)


class IntImage
{
    public:
        IntImage(const IplImage* img)
        {
            assert(img->nChannels == 1 && img->width >= 0 && img->height >= 0);

            width = img->width;
            height = img->height;

            data = new int[width*height];

            calcIntg(img);
        }

        IntImage(int w, int h)
        {
            width = w;
            height = h;

            data = new int[w*h];
        }

        void calcIntg(const IplImage* img)
        {
            assert(img->width == width && img->height == height);

            int sum;
            unsigned char* p;

            sum = 0;
            p = (unsigned char*)img->imageData;
            for(int i = 0; i < width; i ++)
            {
                sum += p[i];
                data[i] = sum;
            }

            sum = 0;
            int k = 0;
            p = (unsigned char*)img->imageData;
            for(int i = 0; i < height; i ++)
            {
                sum += p[0];
                data[k] = sum;
                k += width;
                p += img->widthStep;
            }

            p = (unsigned char*)img->imageData;
            int *ip = data;
            for(int i = 1; i < height; i ++)
            {
                ip += width;
                p += img->widthStep;
                for(int j = 1; j < width; j ++)
                {
                    ip[j] = *(ip+j-1) + *(ip+j-width) - *(ip+j-width-1) + p[j];
                }
            }

        }

        ~IntImage()
        {
            delete[] data;
        }

        int width;
        int height;
        int* data;
        //const IplImage* origImg;
};


class MaskCascadeClassifier : public CascadeClassifier
{
    public:
        CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   const IntImage* ii,
                                   CV_OUT vector<Rect>& objects,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size() );

        CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   CV_OUT vector<Rect>& objects,
                                   vector<int>& rejectLevels,
                                   vector<double>& levelWeights,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size(),
                                   bool outputRejectLevels=false );

    protected:
        virtual bool detectSingleScale( Size winSize, const Mat& image, int stripCount, Size processingRectSize,
                int stripSize, int yStep, double factor, vector<Rect>& candidates,
                vector<int>& rejectLevels, vector<double>& levelWeights, bool outputRejectLevels=false);

        friend class MaskCascadeClassifierInvoker;

        const IntImage* intImg;

};


class MHaarEvaluator : public FeatureEvaluator
{
public:
    struct Feature
    {
        Feature();

        float calc( int offset ) const;
        void updatePtrs( const Mat& sum, double factor);
        bool read( const FileNode& node );

        bool tilted;

        enum { RECT_NUM = 3 };

        struct
        {
            Rect r;
            float weight;
        } rect[RECT_NUM];

        const int* p[RECT_NUM][4];
    };

    MHaarEvaluator();
    virtual ~MHaarEvaluator();

    virtual bool read( const FileNode& node );
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const { return FeatureEvaluator::HAAR; }

    virtual bool setImage(const Mat&, Size origWinSize);
    virtual bool setWindow(Point pt);
    virtual bool setWinSize(Size sz);
    double operator()(int featureIdx) const
    { return featuresPtr[featureIdx].calc(offset) * varianceNormFactor / sqfactor; }
    virtual double calcOrd(int featureIdx) const
    { return (*this)(featureIdx); }

protected:
    Size origWinSize;
    Ptr<vector<Feature> > features;
    Feature* featuresPtr; // optimization
    bool hasTiltedFeatures;

    Mat sum0, sqsum0, tilted0;
    Mat sum, sqsum, tilted;

    Rect normrect;
    const int *p[4];
    const double *pq[4];

    int offset;
    double varianceNormFactor;

protected:
    Size curSize;
    double factor;
    double sqfactor;
};


inline float MHaarEvaluator::Feature :: calc( int _offset ) const
{
    float ret = rect[0].weight * CALC_SUM(p[0], _offset) + rect[1].weight * CALC_SUM(p[1], _offset);

    if( rect[2].weight != 0.0f )
        ret += rect[2].weight * CALC_SUM(p[2], _offset);

    return ret;
}

inline Rect scaleRect(Rect r, double f)
{
    return Rect((int)(f*r.x), (int)(f*r.y), (int)(f*r.width), (int)(f*r.height));
}

inline void MHaarEvaluator::Feature :: updatePtrs( const Mat& _sum, double factor)
{
    const int* ptr = (const int*)_sum.data;
    size_t step = _sum.step/sizeof(ptr[0]);

    Rect rec0 = scaleRect(rect[0].r, factor);
    Rect rec1 = scaleRect(rect[1].r, factor);
    Rect rec2 = scaleRect(rect[2].r, factor);

    if (tilted)
    {
        CV_TILTED_PTRS( p[0][0], p[0][1], p[0][2], p[0][3], ptr, rec0, step );
        CV_TILTED_PTRS( p[1][0], p[1][1], p[1][2], p[1][3], ptr, rec1, step );
        if (rect[2].weight)
            CV_TILTED_PTRS( p[2][0], p[2][1], p[2][2], p[2][3], ptr, rec2, step );
    }
    else
    {
        CV_SUM_PTRS( p[0][0], p[0][1], p[0][2], p[0][3], ptr, rec0, step );
        CV_SUM_PTRS( p[1][0], p[1][1], p[1][2], p[1][3], ptr, rec1, step );
        if (rect[2].weight)
            CV_SUM_PTRS( p[2][0], p[2][1], p[2][2], p[2][3], ptr, rec2, step );
    }
}

#endif
