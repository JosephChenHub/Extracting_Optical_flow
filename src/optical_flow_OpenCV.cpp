#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"


 


#include "CUDA_OpticalFlow.hpp"
#include <sys/types.h>
#include <dirent.h>
#include <sys/time.h> 
#include <time.h> 
#include <sys/stat.h> 
#include <unistd.h>  
#include <omp.h> 

using namespace std;
using namespace cv;

 

 

//#define  TEST_IMAGE	
int searchDir(const std::string & path, std::vector<std::string>&files, const char * ext1 = ".avi" ,const char * ext2 = ".mpg"); //!search all video files
void mkdirs(const char *muldir);


int main(int argc, const char* argv[])
{
 
#ifdef TEST_IMAGE
    string filename1, filename2;
    if (argc < 3)
    {
        cerr << "Usage : " << argv[0] << " <frame0> <frame1>" << endl;
        filename1 = "../data/basketball1.png";
        filename2 = "../data/basketball2.png";
    }
    else
    {
        filename1 = argv[1];
        filename2 = argv[2];
    }

    Mat frame0 = imread(filename1, IMREAD_GRAYSCALE);
    Mat frame1 = imread(filename2, IMREAD_GRAYSCALE);

    CUDA_OpticalFlow m_flow(frame1.size()); 
	Mat dst;
    m_flow.compute(frame0,frame1,dst);
	cv::waitKey();
 
#else
	struct timeval t_start,t_end; 
	long cost_time = 0; 
	static int errorCnt = 0;

    int gpuID = 0;
    int width = 240;
    int height = 240;
    std::string inputPath = "./";
    std::string outputPath = "./";
    std::string rm = "false";
    std::string method = "Brox";

    const char * keys = "{ g|   0    |  gpuID  }"
						"{ w  |   240  |  width  }"
						"{ h  |   240  |  height }"
						"{ i  |   ./   |  input path}" 
						"{ o  |   ./   |  output path}" 
						"{ r  |   false|  remove file}"
						"{ m  |  Brox  |  five methods}";

    cv::CommandLineParser cmd(argc, argv, keys);
    std::cout<<"argc:"<<argc<<std::endl;

    if(argc > 1)
	{
  	 	gpuID = cmd.get<int>("g");
  	 	width = cmd.get<int>("w");
   		height = cmd.get<int>("h");
   		inputPath = cmd.get<std::string>("i");
  	    outputPath = cmd.get<std::string>("o");
		rm = cmd.get<std::string>("r");
		method = cmd.get<std::string>("m");

	}

 
	//! get start time 
	gettimeofday(&t_start, NULL); 
	long start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000; 

	//!delete all video files in the selected path
	std::vector<std::string>files;
    searchDir(inputPath,files);  
	if(rm == "true" || rm == "True" || rm == "1")
	{
		for(int i = 0; i < files.size(); ++i)
		{
			remove(files[i].c_str());
		}
		std::cout<<"all video files is removed!"<<std::endl;
		return 0;
	}
 
    std::cout<<"video files size:"<<files.size()<<" Cores:"<< omp_get_num_procs()<<std::endl;
 	std::cout<<"OPENCV_VERSION:"<<CV_VERSION<<std::endl;
 
	//!extract optical flow using GPU
    setDevice(gpuID); //gpuID

    static int count = 0;

 
    std::cout<<"start processing videos"<<std::endl;
 
	for(int i = 0; i < files.size(); ++i)      
	{
        CUDA_OpticalFlow m_flow(Size(width,height));
		std::string directory(files[i]); 
		directory.erase(directory.end()-4,directory.end());        	//!remove '.avi' or '.mpg'

        count ++;
        float progress =  count*100.0f/files.size(); 
	 	std::cout<<"process video files:"<<files[i]<<"  thread:"<<omp_get_thread_num()<<" progress:"<<progress<<"%"<<std::endl;


		if(outputPath != "./" && !outputPath.empty())
		{
 
			directory.insert(1,"/"+outputPath);
			if(access(directory.c_str(),F_OK) != 0)
			{
				mkdirs(directory.c_str());
			}
		}
		const char * dir = directory.c_str();
		if(access(dir,F_OK) != 0)   //!mkdir
		{
			mkdir(dir,0777);
			std::cout<<"mkdir:"<<dir<<std::endl; 
		} 

 		
		//!now extract frames
 
		cv::VideoCapture capture(files[i]);
 
		if(!capture.isOpened())
		{
			std::cout<<"fail to open video:"<<files[i]<<std::endl;
			errorCnt++;
			continue;
		} 
	    Mat currentFrame,preFrame,frame;
		int cnt = 0;
 		long frames = capture.get(CV_CAP_PROP_FRAME_COUNT);

	    capture.read(frame);  //read the first frame
		cv::resize(frame,currentFrame,cv::Size(width,height));
        cv::cvtColor(currentFrame, currentFrame, CV_BGR2GRAY);


        std::vector<std::vector<double> > mm;
		

		for(int k = 1; k < frames; ++k)
		{
 
			if(!capture.read(frame))
			{
				 
				break;
			}
            currentFrame.copyTo(preFrame);   //pre = current
 
			cv::resize(frame,currentFrame,cv::Size(width,height));
            cv::cvtColor(currentFrame, currentFrame, CV_BGR2GRAY);
			//!name
            std::string str;
		    char  s_cnt[5]; 
		    sprintf(s_cnt,"%04d",cnt);
			str = s_cnt;
            std::string name(dir);  
			name += "/image_" + str;      //！extension is included in writeFlowJpg
 			//!compute
            m_flow.compute(preFrame,currentFrame,method);  //!default method = "Brox"
 
 
#if CV_MAJOR_VERSION >= 3
			 std::vector<double> mm_frame = m_flow.writeFlowJpg(name, m_flow.d_flow); //!write to JPG
#else
			std::vector<double> mm_frame = m_flow.writeFlowJpg(name, m_flow.d_flowx,m_flow.d_flowy); //!write to JPG
#endif

            mm.push_back(mm_frame);   
			cnt ++;
        }

        std::string output_mm(dir);
		output_mm += "/op_minmax.txt";    
		m_flow.writeMM(output_mm, mm);
        std::cout<<"output_MM:"<<output_mm<<std::endl;
		capture.release();
	  }

    files.clear();
	//!end time
	gettimeofday(&t_end, NULL); 
	long end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000; 
	cost_time = end - start;
	if(errorCnt != 0)
	{
	   std::cout<<"error occurred!"<<std::endl;	
	}
	
 	std::cout<<"extracting done!"<<std::endl;
	std::cout<<"cost-time:"<<cost_time/1000.0<<"sec"<<std::endl;
#endif


		
	return 0;
}


  int searchDir(const std::string & path, std::vector<std::string>&files, const char * ext1 ,const char * ext2)  
 {  

    DIR *dp ;  
    struct dirent *dirp ;                         
    const char * c_path = path.c_str();                                 //打开指定目录  
    if( (dp = opendir( c_path )) == NULL )  
    {  
         std::cout<<"No such file or directory:"<<path<<"!"<<std::endl;  
	 return -1;
    }                                  
    while( (dirp = readdir(dp)) != NULL)   				//开始遍历目录
    {                                  
        if(strcmp(dirp->d_name,".")==0  || strcmp(dirp->d_name,"..")==0)  //跳过'.'和'..'两个目录  
            continue;  
        int size = strlen(dirp->d_name); 
		if(dirp->d_type == 4) 							//目录继续
		{
			std::string name(dirp->d_name);
			std::string child(path);
			if(strcmp(c_path,"./") != 0)  //path != "./",then add "/"
			{
			  child += "/";
 			}
			child += name;
		 	searchDir(child,files,ext1,ext2);
		}  
		else                           //file
		{
       	 if(strcmp( ( dirp->d_name + (size - 4) ) , ext1) != 0 && strcmp( ( dirp->d_name + (size - 4) ) , ext2) != 0 )   //只存取xxx扩展名的文件名
            	   continue; 
		 else
		 {
			std::string name(dirp->d_name);
			std::string child(path);
			if(strcmp(c_path,"./") != 0)  //path != "./",then add "/"
			{
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

void mkdirs(const char *muldir) 
{
    int i,len;
    char str[512];    
    strncpy(str, muldir, 512);
    len=strlen(str);
    for( i=0; i<len; i++ )
    {
        if( str[i]=='/' )
        {
            str[i] = '\0';
            if( access(str,0)!=0 )
            {
                mkdir( str, 0777 );
            }
            str[i]='/';
        }
    }
    if( len>0 && access(str,0)!=0 )
    {
        mkdir( str, 0777 );
    }
    return;
}
