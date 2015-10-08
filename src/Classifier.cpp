#include "Classifier.h"

#include <iostream>
using namespace std;

#include "config.h"

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };


class MaskCascadeClassifierInvoker : public ParallelLoopBody
{
public:
    MaskCascadeClassifierInvoker(CvRect oRect, CvRect cRect, MaskCascadeClassifier& _cc, Size _sz1, int _stripSize, int _yStep, double _factor,
        vector<Rect>& _vec, vector<int>& _levels, vector<double>& _weights, bool outputLevels, const Mat& _mask, Mutex* _mtx)
    {
        classifier = &_cc;
        processingRectSize = _sz1;
        stripSize = _stripSize;
        yStep = _yStep;
        scalingFactor = _factor;
        rectangles = &_vec;
        rejectLevels = outputLevels ? &_levels : 0;
        levelWeights = outputLevels ? &_weights : 0;
        mask = _mask;
        mtx = _mtx;
		originRect = oRect;
		currentRect = cRect;
    }

    void operator()(const Range& range) const
    {
        Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();

        Size winSize(cvRound(classifier->data.origWinSize.width * scalingFactor), cvRound(classifier->data.origWinSize.height * scalingFactor));

		int y1, y2, x1, x2;

		x1 = currentRect.x;
		x2 = max(0, currentRect.x + currentRect.width - classifier->data.origWinSize.width);
		y1 = range.start * stripSize + currentRect.y;
		y2 = min(range.end * stripSize + currentRect.y, currentRect.y + processingRectSize.height - classifier->data.origWinSize.height);


        for( int y = y1; y < y2; y += yStep )
        {
            for( int x = x1; x < x2; x += yStep )
            {
                if ( (!mask.empty()) && (mask.at<uchar>(Point(x,y))==0)) {
                    continue;
                }

                if(opt)
                {
                    const IntImage* intImg = classifier->intImg;
                    int width = intImg->width;
                    int height = intImg->height;

                    int origX = x * scalingFactor;
                    int origY = y * scalingFactor;
                    int origX2 = origX + winSize.width;
                    int origY2 = origY + winSize.height;
					if (origX2 > intImg->width || origY2 > intImg->height)
						continue;

                    int area = winSize.width * winSize.height;
                    int integ = intImg->data[origY2*width+origX2] 
                        - intImg->data[origY*width+origX2] 
                        - intImg->data[origY2*width+origX] 
                        + intImg->data[origY*width+origX];
                    if(integ < area / opt_factor)
                        continue;
                }


                double gypWeight;
                int result = classifier->runAt(evaluator, Point(x, y), gypWeight);

#if defined (LOG_CASCADE_STATISTIC)

                logger.setPoint(Point(x, y), result);
#endif
                if( rejectLevels )
                {
                    if( result == 1 )
                        result =  -(int)classifier->data.stages.size();
                    if( classifier->data.stages.size() + result < 4 )
                    {
                        mtx->lock();
                        rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor), winSize.width, winSize.height));
                        rejectLevels->push_back(-result);
                        levelWeights->push_back(gypWeight);
                        mtx->unlock();
                    }
                }
                else if( result > 0 )
                {
                    mtx->lock();
                    rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor),
                                               winSize.width, winSize.height));
                    mtx->unlock();
                }
                if( result == 0 )
                    x += yStep;
            }
        }
    }

    MaskCascadeClassifier* classifier;
    vector<Rect>* rectangles;
    Size processingRectSize;
    int stripSize, yStep;
    double scalingFactor;
    vector<int> *rejectLevels;
    vector<double> *levelWeights;
    Mat mask;
    Mutex* mtx;
	CvRect originRect;
	CvRect currentRect;
};



bool MaskCascadeClassifier::detectSingleScale(CvRect oRect, CvRect cRect, const Mat& image, int stripStart, int stripEnd, Size processingRectSize,
                                           int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                           vector<int>& levels, vector<double>& weights, bool outputRejectLevels )
{
    if( !featureEvaluator->setImage( image, data.origWinSize ) )
        return false;

#if defined (LOG_CASCADE_STATISTIC)
    logger.setImage(image);
#endif

    Mat currentMask;
    if (!maskGenerator.empty()) {
        currentMask=maskGenerator->generateMask(image);
    }

    vector<Rect> candidatesVector;
    vector<int> rejectLevels;
    vector<double> levelWeights;
    Mutex mtx;
    if( outputRejectLevels )
    {
        parallel_for_(Range(stripStart, stripEnd), MaskCascadeClassifierInvoker(oRect, cRect, *this, processingRectSize, stripSize, yStep, factor,
            candidatesVector, rejectLevels, levelWeights, true, currentMask, &mtx));
        levels.insert( levels.end(), rejectLevels.begin(), rejectLevels.end() );
        weights.insert( weights.end(), levelWeights.begin(), levelWeights.end() );
    }
    else
    {
		parallel_for_(Range(stripStart, stripEnd), MaskCascadeClassifierInvoker(oRect, cRect, *this, processingRectSize, stripSize, yStep, factor,
            candidatesVector, rejectLevels, levelWeights, false, currentMask, &mtx));
    }
    candidates.insert( candidates.end(), candidatesVector.begin(), candidatesVector.end() );

#if defined (LOG_CASCADE_STATISTIC)
    logger.write();
#endif

    return true;
}


void MaskCascadeClassifier::detectMultiScale(CvRect ori, const Mat& image, const IntImage* ii, vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize)
{
    intImg = ii;
    vector<int> fakeLevels;
    vector<double> fakeWeights;
    detectMultiScale(ori, image, objects, fakeLevels, fakeWeights, scaleFactor,
        minNeighbors, flags, minObjectSize, maxObjectSize, false );
}

void MaskCascadeClassifier::detectMultiScale(CvRect ori, const Mat& image, vector<Rect>& objects,
                                          vector<int>& rejectLevels,
                                          vector<double>& levelWeights,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels )
{
    const double GROUP_EPS = 0.2;

    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if( empty() )
        return;

    if( isOldFormatCascade() )
    {
        MemStorage storage(cvCreateMemStorage(0));
        CvMat _image = image;
        CvSeq* _objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
                                              minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
        vector<CvAvgComp> vecAvgComp;
        Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
        objects.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
        return;
    }

    objects.clear();

    if (!maskGenerator.empty()) {
        maskGenerator->initializeMask(image);
    }


    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = image.size();

    Mat grayImage = image;
    if( grayImage.channels() > 1 )
    {
        Mat temp;
        cvtColor(grayImage, temp, CV_BGR2GRAY);
        grayImage = temp;
    }

    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
    vector<Rect> candidates;

    for( double factor = 1; ; factor *= scaleFactor )
    {
        Size originalWindowSize = getOriginalWindowSize();

        Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        Size scaledImageSize( cvRound( grayImage.cols/factor ), cvRound( grayImage.rows/factor ) );
        
		CvRect cRect;
		if (opt)
		{
			cRect.x = min(int(ori.x / factor), scaledImageSize.width - 1);
			cRect.y = min(int(ori.y / factor), scaledImageSize.height - 1);
			cRect.width = min(int(ori.width / factor), scaledImageSize.width);
			cRect.height = min(int(ori.height / factor), scaledImageSize.height);
		}
		else
		{
			cRect = cvRect(0, 0, scaledImageSize.width, scaledImageSize.height);
		}
		
		
        Size processingRectSize(cRect.width - originalWindowSize.width, cRect.height - originalWindowSize.height );

        if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
            break;
        if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height )
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;

        Mat scaledImage(scaledImageSize, CV_8U, imageBuffer.data );
        resize(grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );

        int yStep;
        if( getFeatureType() == cv::FeatureEvaluator::HOG )
        {
            yStep = 4;
        }
        else
        {
            yStep = factor > 2. ? 1 : 2;
        }

        int stripCount, stripSize;

        const int PTS_PER_THREAD = 1000;
		
        stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
        stripCount = std::min(std::max(stripCount, 1), 100);
        stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;

        if( !detectSingleScale(ori, cRect, scaledImage, 0, stripCount, processingRectSize, stripSize, yStep, factor, candidates,
            rejectLevels, levelWeights, outputRejectLevels ) )
            break;
    }


    objects.resize(candidates.size());
    std::copy(candidates.begin(), candidates.end(), objects.begin());

    if( outputRejectLevels )
    {
        groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
    }
    else
    {
        groupRectangles( objects, minNeighbors, GROUP_EPS );
    }
}
