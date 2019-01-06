#ifndef __MY_UTILS_HPP
#define __MY_UTILS_HPP


#include <sys/types.h>
#include <dirent.h>
#include <sys/time.h> 
#include <sys/stat.h> 
#include <unistd.h>  
#include <vector>
#include <string>


int globFiles(const std::string & path, std::vector<std::string>&files, const char * ext1 = ".avi" ,const char * ext2 = ".mpg", const char *ext3=".mp4"); //!search all video files
void makedirs(const char *muldir);




#endif
