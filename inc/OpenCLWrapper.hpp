#ifndef OPENCLWRAPPER_HPP
# define OPENCLWRAPPER_HPP

# ifdef __APPLE__
#  include <OpenCL/opencl.h>
#  include <OpenGL/CGLTypes.h>
#  include <OpenGL/CGLCurrent.h>
# else
#  include <CL/cl.h>
#  include <CL/cl_gl.h>
#  include <GL/glx.h>
# endif

# include <GLFW/glfw3.h>
# include <vector>
# include <string>

# include "Utils.hpp"

class OpenCLWrapper
{
public:
	bool						initialized;

	cl_uint						num_entries;
	cl_platform_id				platformID;
	cl_uint						num_platforms;
	cl_device_id				clDeviceId;
	cl_context					clContext;
	cl_command_queue			clCommands;
	std::vector<cl_program>		clPrograms;
	std::vector<cl_kernel>		clKernels;
	std::vector<cl_int>			local;
	size_t						programNumber;

	OpenCLWrapper();
	~OpenCLWrapper();

	cl_int					init(void);
	cl_int					initKernels(std::vector<std::string> const &kernelFiles,
										std::vector<std::string> const &kernelNames,
										std::string const &options);
	cl_int					getOpenCLInfo(void);
private:
	OpenCLWrapper(OpenCLWrapper const &src);

	cl_int					cleanDeviceMemory(void);
};

#endif