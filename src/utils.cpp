#include "utils.hpp"
#include <iostream>
#include <string.h>

int globFiles(const std::string & path, std::vector<std::string>&files, const char * ext1,\
        const char * ext2, const char * ext3) {  
    DIR *dp ;  
    struct dirent *dirp ;                         
    const char * c_path = path.c_str();                               
    if( (dp = opendir( c_path )) == NULL )  {  
        std::cout<<"No such file or directory:"<<path<<"!"<<std::endl;  
	    return -1;
    }                                  
    while( (dirp = readdir(dp)) != NULL)  {                                  
        if(strcmp(dirp->d_name,".")==0  || strcmp(dirp->d_name,"..")==0)
            continue;  
        int size = strlen(dirp->d_name); 
		if(dirp->d_type == 4)  { //!directory
			std::string name(dirp->d_name);
			std::string child(path);
			if(strcmp(c_path,"./") != 0)  {
			  child += "/";
 			}
			child += name;
		 	globFiles(child,files,ext1,ext2);
		}  
		else { //! file
       	 if(strcmp(dirp->d_name + (size - 4), ext1) != 0 && strcmp(dirp->d_name + (size - 4), ext2) != 0 && strcmp(dirp->d_name +(size-4), ext3) != 0)
            	   continue; 
		 else {
			std::string name(dirp->d_name);
			std::string child(path);
			if(strcmp(c_path,"./") != 0)  {
			  child += "/";
 			}
			child += name;
			files.push_back(child);
		 }
	    }  
    }  
    closedir(dp);  
    return 0;
} 

void makedirs(const char *muldir) {
    char str[512];    
    strncpy(str, muldir, 512);
    int len=strlen(str);
    for(int i=0; i<len; i++ ) {
        if(str[i]=='/') {
            str[i] = '\0';
            if( access(str,0)!=0 ) {
                mkdir( str, 0777 );
            }
            str[i]='/';
        }
    }
    if( len>0 && access(str,0)!=0 ) {
        mkdir( str, 0777 );
    }
}
