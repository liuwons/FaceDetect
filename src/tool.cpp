#include "tool.h"
#include "config.h"

#include <iostream>
using namespace std;

bool parse_param(int argc, char** argv, map<string, string>& params)
{
    params["mode"] = MODE_IMAGE;
    params["wframe"] = "no";
    params["debug"] = "no";
    params["log"] = ".";
    params["fps"] = "10";
    params["fbuf"] = "10";
    params["classfier"] = "normal";
    params["opt_factor"] = "3";
	params["light_opt"] = "no";
	params["save_image"] = "no";

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
        else if(!strcmp(argv[i], "--classifier"))
        {
            params["classifier"] = argv[i+1];
            if(!strcmp(argv[i+1], "opt"))
                opt = true;
            i++;
        }
        else if(!strcmp(argv[i], "--opt_factor"))
        {
            params["opt_factor"] = argv[i+1];
            opt_factor = atoi(argv[i+1]);
            cout << "opt_factor:" << opt_factor << endl;
            i++;
        }
		else if (!strcmp(argv[i], "--light_opt"))
		{
			params["light_opt"] = "yes";
		}
		else if (!strcmp(argv[i], "--save_image"))
		{
			params["save_image"] = "yes";
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
