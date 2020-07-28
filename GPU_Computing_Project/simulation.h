#pragma once
#include <cuda_runtime.h>
#include <array>
#include <vector>


struct GLtextures {
	unsigned vector_field;
	unsigned color_output;
};

struct CudaGLResources {
	cudaGraphicsResource * vector_field;
	cudaGraphicsResource * color_output;
};

struct SimulationState{
	GLtextures gl_textures;
	CudaGLResources cuda_resources;
	cudaArray* vector_field;
	cudaArray* dye_texture;
	std::vector<float>* cpu_vector_field;
	std::vector<float>* cpu_color_field;
};

struct SplatData {
	bool splatting;
	float splat_color[3] = {0.0f, 1.0f, 0.0f};
	float splat_radius = 0.025f;
	float splat_posx, splat_posy;
	float splat_impuls_X, splat_impuls_Y;
};

class SimulationData{
public:
	unsigned sim_width;
	unsigned sim_height;
	unsigned dye_height;
	unsigned dye_width;
	int block_width;
	int block_heigth;
	std::array<SimulationState, 2> simstates;
	double dt;
	//float base_pressure = 0.0;
	int iterations = 20;
	float density_diffusion = 0.005;
	float viscosity = 0.005;
	// splat new transportable dye
	SplatData splat;
	auto prev() {return &simstates[0];}
	auto next() { return &simstates[1]; }
	void swapStates() {
		SimulationState tmp = this->simstates[0];
		this->simstates[0] = this->simstates[1];
		this->simstates[1] = tmp;
	}
};

void launchSimKernels(SimulationData &data);
void launchCPUImpl(SimulationData &data);


inline void check(cudaError_t result, char const *const func, int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at line:%d code=%d(%s) \"%s\" \n", line,
			static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		while (getchar() != 10);
		exit(EXIT_FAILURE);
	}
}
#define checkCudaErrors(val) check((val), #val, __LINE__)
