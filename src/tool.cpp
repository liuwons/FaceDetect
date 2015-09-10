#include "tool.h"

bool parse_param(int argc, char** argv, map<string, string>& params)
{
    params["mode"] = MODE_IMAGE;
    params["wframe"] = "no";
    params["debug"] = "no";
    params["log"] = ".";
    params["fps"] = 10;
    params["fbuf"] = 10;

    for(int i = 1; i < argc; i ++)
    {
        if(!strcmp(argv[i], "--mode"))
        {
            if(!strcmp(argv[i+1], "video"))
            {
                params["mode"] = MODE_VIDEO;
                i++;
            }
            else if(!strcmp(argv[i+1], "image"))
            {
                params["mode"] = MODE_IMAGE;
                i++;
            }
            else
            {
                return false;
            }
        }
        else if(!strcmp(argv[i], "--wframe"))
        {
            params["wframe"] = "yes";
        }
        else if(!strcmp(argv[i], "--debug"))
        {
            params["debug"] = "yes";
        }
        else if(!strcmp(argv[i], "--log"))
        {
            params["log"] = argv[i+1];
            i++;
        }
        else if(!strcmp(argv[i], "--fps"))
        {
            params["fps"] = argv[i+1];
            i++;
        }
        else if(!strcmp(argv[i], "--fbuf"))
        {
            params["fbuf"] = argv[i+1];
            i++;
        }
        else
        {
            params["file"] = argv[i];
        }
    }
    return true;
}

void setImageVal(IplImage* img, unsigned char val)
{
    unsigned char* p = (unsigned char*)img->imageData;
    for(int i = 0; i < img->height; i ++)
    {
        memset(p, val, img->widthStep);
        p += img->widthStep;
    }
}
