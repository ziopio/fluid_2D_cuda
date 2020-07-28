#pragma once

#include <GL/gl3w.h>
#include "simulation.h"

namespace gui {
	extern bool stop_time;
	extern bool debug_flag;
	extern bool use_CPU;
	extern bool reload_sim_buffers;
	extern bool reload_dye_buffers;
	extern bool cuda_supported;
	extern std::string gpu_name;
	extern unsigned w_width, w_height;
	// Create a GLSL program object from vertex and fragment shader files
	unsigned initShader(const char * vShaderFile, const char * fShaderFile);
	unsigned allocateOpenGLTexture(unsigned width, unsigned height, unsigned init_value);

	void loadDataToTexture(unsigned textureID, float * data, unsigned w, unsigned h);

	void initGui(unsigned width, unsigned height, void (* mouseButtonCallback)(int,int,int), void(*cursoPosCallback)(double,double));
	bool shouldCloseApp();
	void preRender();


	// in milliseconds
	double getDeltaTime();
	void renderGUIOverlay(SimulationData &sim);
	void presentToScreen();
	void cleanUp();

}

