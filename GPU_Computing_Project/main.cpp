#include "gui.h"
#include <iostream>
#include <array>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include "simulation.h"

static bool cuda_registered_gl_textures = false;

static GLuint shader; // OpenGL shader
static GLuint dummyVAO; // yeah empty and dumb
static SimulationData sim; // Our simulation data

// controll
static bool input_ink;
static double last_xpos, last_ypos;

bool checkCudaDevice() {
	int n = 0;
	if (cudaGetDeviceCount(&n) != cudaError::cudaSuccess) {
		return false;
	}
	for (int i = 0; i < n; i++) {
		cudaDeviceProp properties;
		checkCudaErrors(cudaGetDeviceProperties(&properties, i));
		printf("Device Number: %d\n", i);
		printf("Device name: %s\n", properties.name);
	}
	return n;
}

void mouseButtonCallback(int button, int action, int mods) {
	input_ink = action == 0 ? false : true;
	//std::cout << "input click" << std::endl;
}

void cursoPosCallback(double xpos, double ypos) {
	//screen mapping to 0-1
	xpos = xpos / gui::w_width;
	ypos = ypos / gui::w_height;
	if (input_ink) {
		sim.splat.splatting = true;
		sim.splat.splat_posx = last_xpos;
		sim.splat.splat_posy = last_ypos;
		sim.splat.splat_impuls_X = xpos - last_xpos;
		sim.splat.splat_impuls_Y = ypos - last_ypos;
	}
	last_xpos = xpos;
	last_ypos = ypos;
	//std::cout << "input pos: " << xpos << " " << ypos << std::endl;
}

void registerCudaResources(SimulationState &state) {
	checkCudaErrors(cudaGraphicsGLRegisterImage(
		&state.cuda_resources.vector_field,
		state.gl_textures.vector_field,
		GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsSurfaceLoadStore));
	checkCudaErrors(cudaGraphicsGLRegisterImage(
		&state.cuda_resources.color_output,
		state.gl_textures.color_output,
		GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsSurfaceLoadStore));
}
void unRegisterCudaResources(SimulationState &state) {
	checkCudaErrors(cudaGraphicsUnregisterResource(state.cuda_resources.vector_field));
	checkCudaErrors(cudaGraphicsUnregisterResource(state.cuda_resources.color_output));
}

void setup() {
	// Each State is saved in a texture 2 times and registered as GL resources in CUDA
	for (auto &state : sim.simstates)
	{
		state.gl_textures.vector_field = gui::allocateOpenGLTexture(sim.sim_width, sim.sim_height, 128);
		state.gl_textures.color_output = gui::allocateOpenGLTexture(sim.dye_width, sim.dye_height, 0);
		
		if (gui::cuda_supported) {
			registerCudaResources(state);
		}
		
		// CPU bufers to enable serial implementation
		state.cpu_color_field = new std::vector<float>(4*sim.dye_width*sim.dye_height, 0.0f);
		state.cpu_vector_field = new std::vector<float>(4*sim.sim_width*sim.sim_height, 0.5f);
	}
	if (gui::cuda_supported) {
		cuda_registered_gl_textures = true;
	}
	gui::reload_sim_buffers = false;
	gui::reload_dye_buffers = false;
}


void cleanUP() {
	for (auto &state : sim.simstates)
	{
		if (cuda_registered_gl_textures) {
			unRegisterCudaResources(state);
		}
		glDeleteTextures(1, &state.gl_textures.vector_field);
		glDeleteTextures(1, &state.gl_textures.color_output);
		// CPU bufers to enable serial implementations
		delete state.cpu_color_field;
		delete state.cpu_vector_field;
	}
}


int main(int, char**)
{
	gui::initGui(1200, 700, mouseButtonCallback, cursoPosCallback);
	// custom shader
	shader = gui::initShader("post_process.vert", "texture_sampling.frag");
	glGenVertexArrays(1, &dummyVAO);
	// check cuda
	gui::cuda_supported = checkCudaDevice();

	sim.block_width = 16;
	sim.block_heigth = 16;
	//Simulation Buffer dimension
	sim.sim_width = gui::cuda_supported ? 256 : 64;
	sim.sim_height = sim.sim_width;
	// Dye texture dimensions
	sim.dye_width = gui::cuda_supported ? 1024 : 256;
	sim.dye_height = sim.dye_width;

	setup();

    // Main loop
    while (!gui::shouldCloseApp())
    {
		gui::preRender();
		// Check if user changed resolution
		if (gui::reload_sim_buffers || gui::reload_dye_buffers) {
			cleanUP();
			setup();
		}

		// We swap our states so that we use the previous result as input 
		// while the old as writable output
		sim.swapStates();
		sim.dt = gui::getDeltaTime() / 10.0;
		sim.dt = gui::stop_time ? 0 : sim.dt;
		if (gui::use_CPU || !gui::cuda_supported) 
		{
			if (cuda_registered_gl_textures) {
				for (auto &state : sim.simstates) {
					unRegisterCudaResources(state);
				}
				cuda_registered_gl_textures = false;
			}
			launchCPUImpl(sim);
			gui::loadDataToTexture(sim.next()->gl_textures.color_output, sim.next()->cpu_color_field->data(), sim.dye_width, sim.dye_height);
			gui::loadDataToTexture(sim.next()->gl_textures.vector_field, sim.next()->cpu_vector_field->data(), sim.sim_width, sim.sim_height);
		}
		else 
//-----------------------------------CUDA-ZONE-------------------------------------
		{
			for (auto &state : sim.simstates) // Mapping
			{
				if (!cuda_registered_gl_textures) {
					registerCudaResources(state);
				}
				// Divergence field
				checkCudaErrors(cudaGraphicsMapResources(1, &state.cuda_resources.vector_field, 0));
				checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
					&state.vector_field, state.cuda_resources.vector_field, 0, 0));
				// Color output
				checkCudaErrors(cudaGraphicsMapResources(1, &state.cuda_resources.color_output, 0));
				checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
					&state.dye_texture, state.cuda_resources.color_output, 0, 0));

			}
			cuda_registered_gl_textures = true;

			// KERNEL LAUNCH
			launchSimKernels(sim);

			for (auto &sim : sim.simstates) // Un-Mapping
			{
				checkCudaErrors(cudaGraphicsUnmapResources(1, &sim.cuda_resources.vector_field, 0));
				checkCudaErrors(cudaGraphicsUnmapResources(1, &sim.cuda_resources.color_output, 0));
			}
		}
		sim.splat.splatting = false;
//----------------------------------OPENGL-ZONE-------------------------------------
		glUseProgram(shader);
		GLuint d_loc = glGetUniformLocation(shader, "debug_flag");
		GLuint v_loc = glGetUniformLocation(shader, "velocity_sampler");
		GLuint c_loc = glGetUniformLocation(shader, "color_sampler");
		glUniform1i(d_loc, gui::debug_flag);
		glUniform1i(v_loc, 0);
		glUniform1i(c_loc, 1);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, sim.next()->gl_textures.vector_field);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, sim.next()->gl_textures.color_output);
		glBindVertexArray(dummyVAO);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		gui::renderGUIOverlay(sim);
		gui::presentToScreen();
    }
//-------------------------------------CLEAN-UP----------------------------------
	cleanUP();

	glDeleteVertexArrays(1, &dummyVAO);
	glDeleteProgram(shader);
	gui::cleanUp();

    return 0;
}
