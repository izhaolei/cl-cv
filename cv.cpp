#include <iostream>
#include <CL/cl.hpp>
#include <vector>

//#include <sstream>
//#include <ostream>
//#include <stdio.h>
//#include <stdlib.h>
#include <fstream>
#include<string>
#include <opencv2/opencv.hpp>
//#include <opencv2/ocl/ocl.hpp>
using namespace std;
std::string readFile(std::string fileName)
{
	std::ifstream t(fileName);
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return str;
}

int main(int arg, char* args[])
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
	{
		std::cout << "No OpenCL platforms found" << std::endl;//This means you do not have an OpenCL compatible platform on your system.
		exit(1);
	}
	cl_context_properties properties[] =
	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, properties);
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	//std::vector<cl::Device> devices;
	//platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	//cl::Device device = devices[0];
	//std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	//std::cout << "Using platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
	//cl::Context context({ device });
	
	cv::Mat image = cv::imread("C:/Users/zhaolei/Desktop/²¶»ñ.JPG", CV_LOAD_IMAGE_COLOR);
	cv::Mat imageRGBA;
	cv::cvtColor(image, imageRGBA, CV_RGB2RGBA);
	cv::namedWindow("he");

	unsigned char *buffer = reinterpret_cast<unsigned char *>(imageRGBA.data);
	imageRGBA.data = buffer;
	cv::imshow("he", imageRGBA);
	size_t width = imageRGBA.cols;
	size_t height = imageRGBA.rows;
	cl::Image2D clImage(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
		width,
		height,
		0,
		buffer);
	cl::Image2D out(context,
		CL_MEM_READ_WRITE,
		cl::ImageFormat(CL_RGBA, CL_UNORM_INT8), 
		width, 
		height);
	cl::Program::Sources sources;
	std::string copy =
		"const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;"

		"__kernel void copy(read_only image2d_t src, write_only image2d_t dst)"
		"{"
		"int x = (int)get_global_id(0); int y = (int)get_global_id(1);"
		"if (x >= get_image_width(src) || y >= get_image_height(src)) return;"

		"	float4 p00 = read_imagef(src, sampler, (int2)(x - 1, y - 1));"
		"float4 p10 = read_imagef(src, sampler, (int2)(x, y - 1));"
		"float4 p20 = read_imagef(src, sampler, (int2)(x + 1, y - 1));"

		"	float4 p01 = read_imagef(src, sampler, (int2)(x - 1, y));"
		"	float4 p21 = read_imagef(src, sampler, (int2)(x + 1, y));"

		"	float4 p02 = read_imagef(src, sampler, (int2)(x - 1, y + 1));"
		"	float4 p12 = read_imagef(src, sampler, (int2)(x, y + 1));"

		"	float4 p22 = read_imagef(src, sampler, (int2)(x + 1, y + 1));"
		"	float3 gx = -p00.xyz + p20.xyz + 2.0f * (p21.xyz - p01.xyz) - p02.xyz + p22.xyz;"
		"	float3 gy = -p00.xyz - p20.xyz + 2.0f * (p12.xyz - p10.xyz) + p02.xyz + p22.xyz;"
		"	float3 norm = gx * gx + gy * gy;"
		"	float3 g = native_sqrt(norm.x + norm.y + norm.z);"
		"	write_imagef(dst, (int2)(x, y), (float4)(g.x, g.y, g.z, 1.0f));"
		"}";
	//std::string read =
	//	"void kernel read(__read_only image2d_t in,   int w,  int h)"
	//	"{"
	//	"w=get_image_width(in);"
	//	"h=get_image_height(in);\n"
	//	"}";
	//Add your program source
	sources.push_back({ copy.c_str(), copy.length() });

	//Create your OpenCL program and build it.
	cl::Program program(context, sources);
	if (program.build({ devices }) != CL_SUCCESS)
	{
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;//print the build log to find any issues with your source
		exit(1);//Quit if your program doesn't compile
	}
	//set the kernel arguments
	cl::Kernel kernelcopy(program, "copy");
	kernelcopy.setArg(0, clImage);
	kernelcopy.setArg(1, out);
	cl::size_t<3> origin;
	cl::size_t<3> size;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;
	size[0] = width;
	size[1] = height;
	size[2] = 1;
	cv::Mat as(height, width, CV_8UC4);
	cv::namedWindow("he1");
	//...
	
	cl::CommandQueue queue(context, devices[0], 0, NULL);
	
	//execute kernel
	//have a two dimensional global range of the width and height of our image so we can go through all of the pixels of the image
	queue.enqueueNDRangeKernel(kernelcopy, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
	
	//wait for kernel to finish
	queue.finish();
	queue.enqueueReadImage(out, CL_TRUE, origin, size, 0, 0, as.data);

	//CL_TRUE means that it waits for the entire image to be copied before continuing
	//queue.enqueueReadImage(clImage, CL_TRUE, origin, size, 0, 0, as.data);
	cv::imshow("he1", as);

	cv::waitKey(10000);
	return 0;
}