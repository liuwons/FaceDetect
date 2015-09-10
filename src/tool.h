#ifndef TOOL_H
#define TOOL_H

#include <map>
#include <string>

#include "cv_header.h"

using std::map;
using std::string;

const static string MODE_IMAGE = "MODE_IMAGE";
const static string MODE_VIDEO = "MODE_VIDEO";

bool parse_param(int argc, char** argv, map<string, string>& params);
void setImageVal(IplImage* img, unsigned char val);

#endif
