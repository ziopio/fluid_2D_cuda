#include "simulation.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math.h"
//#include <cooperative_groups.h>

__device__ float clampTo_0_1(float val) {
	if (val < 0.f) val = 0;
	if (val > 1.0f) val = 1.f;
	return val;
}

__device__ float3 bilerp(cudaSurfaceObject_t source, float2 coord, unsigned width, unsigned height) 
{
	if (ceil(coord.x) == floor(coord.x)) {
		coord.x -= 0.0001;
	}
	if (ceil(coord.y) == floor(coord.y)) {
		coord.y -= 0.0001;
	}
	// Getting 4 adiacent pixels positions
	int2 bot_right = make_int2(ceil(coord.x), ceil(coord.y));
	int2 bot_left = make_int2(floor(coord.x), ceil(coord.y));
	int2 top_right = make_int2(ceil(coord.x), floor(coord.y));
	int2 top_left = make_int2(floor(coord.x), floor(coord.y));
	// Getting the 4 pixels values
	float4 color_br, color_bl, color_tr, color_tl;
	surf2Dread(&color_br, source, bot_right.x * sizeof(float4), bot_right.y, cudaBoundaryModeClamp);
	surf2Dread(&color_bl, source, bot_left.x * sizeof(float4), bot_left.y, cudaBoundaryModeClamp);
	surf2Dread(&color_tl, source, top_left.x * sizeof(float4), top_left.y, cudaBoundaryModeClamp);
	surf2Dread(&color_tr, source, top_right.x * sizeof(float4), top_right.y, cudaBoundaryModeClamp);

	//horizontal interpolation for the BOTTOM pixels
	float Xdiff_left = coord.x - (float)bot_left.x;
	float Xdiff_right = (float)bot_right.x - coord.x;
	float Hx1 = color_bl.x * Xdiff_right  + color_br.x * Xdiff_left ; // x comp
	float Hy1 = color_bl.y * Xdiff_right  + color_br.y * Xdiff_left ; // y comp
	float Hz1 = color_bl.z * Xdiff_right + color_br.z * Xdiff_left; // z comp

	//horizontal interpolation for the TOP pixels
	Xdiff_left = coord.x - (float)top_left.x;
	Xdiff_right = (float)top_right.x - coord.x;
	float Hx2 = color_tl.x * Xdiff_right + color_tr.x * Xdiff_left; // x comp
	float Hy2 = color_tl.y * Xdiff_right + color_tr.y * Xdiff_left; // y comp
	float Hz2 = color_tl.z * Xdiff_right + color_tr.z * Xdiff_left; // z comp

	// vertical interpolation: note Y starts from the top of the image
	float Ydiff_top = coord.y - (float)top_right.y;
	float Ydiff_bot = (float)bot_right.y - coord.y;
	float X = Hx1 * Ydiff_top + Hx2 * Ydiff_bot;
	float Y = Hy1 * Ydiff_top + Hy2 * Ydiff_bot;
	float Z = Hz1 * Ydiff_top + Hz2 * Ydiff_bot;

	return make_float3(X,Y,Z);
}

__global__ void advectionStep(
	cudaSurfaceObject_t source, 
	cudaSurfaceObject_t target, 
	unsigned width, unsigned height, SimulationData data)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		float2 old_pos;
		float4 local_val = make_float4(0, 0, 0, 0);
		surf2Dread(&local_val, source, x * sizeof(float4), y, cudaBoundaryModeClamp);
		// follow the velocity field "back in time"
		// dt is in milliseconds, color channels are shifted with *2-1 to indicate negative directions
		old_pos.x = (float)x - data.dt * (local_val.x * 2 - 1);
		old_pos.y = (float)y - data.dt * (local_val.y * 2 - 1);
		// interpolate and write to the output
		float3 newV = bilerp(source, old_pos, width, height);
		float decay = 1.f + data.viscosity * data.dt;
		newV.x = ((newV.x * 2 - 1) / decay + 1) / 2;
		newV.y = ((newV.y * 2 - 1) / decay + 1) / 2;
		newV.z = ((newV.z * 2 - 1) / decay + 1) / 2;
		surf2Dwrite(make_float4(newV.x, newV.y, local_val.z, 1.0), target, x * sizeof(float4), y);
	}
}

__global__ void forceApplicationStep( cudaSurfaceObject_t target, unsigned width, unsigned height, SplatData splat)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		float forceX = splat.splat_impuls_X * width;
		float forceY = splat.splat_impuls_Y * height;
		float influence = 1.0f * exp(-((x - splat.splat_posx * width)*(x - splat.splat_posx * width) +
			(y - splat.splat_posy * height)*(y - splat.splat_posy * height)) /
			((splat.splat_radius * height)*(splat.splat_radius * height)*2));
		//printf("x = %u, splat.splat_posx = %f, influence = %f\n",x, splat.splat_posx, influence);

		influence = clampTo_0_1(influence);
		float4 current_val;
		surf2Dread(&current_val, target, x * sizeof(float4), y, cudaSurfaceBoundaryMode::cudaBoundaryModeClamp);
		current_val.x += forceX * influence;
		current_val.y += forceY * influence;
		current_val.x = clampTo_0_1(current_val.x);
		current_val.y = clampTo_0_1(current_val.y);
		//printf("current_val.x = %f, current_val.y = %f, influence = %f\n", current_val.x, current_val.y, influence);
		surf2Dwrite(current_val, target, x * sizeof(float4), y);
	}
}

__global__ void divergence(cudaSurfaceObject_t source, cudaSurfaceObject_t target, unsigned width, unsigned height)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > 0 && x < width -1 && y < height -1 && y > 0) {
		// Getting 4 adiacent pixels positions
		int2 right = make_int2(x + 1, y);
		int2 left = make_int2(x - 1, y);
		int2 top = make_int2(x, y - 1);
		int2 bot = make_int2(x, y + 1);
		// Getting the 4 pixels values
		float4 val_top, val_bot, val_left, val_rigth;
		surf2Dread(&val_top, source, top.x * sizeof(float4), top.y, cudaBoundaryModeClamp);
		surf2Dread(&val_bot, source, bot.x * sizeof(float4), bot.y, cudaBoundaryModeClamp);
		surf2Dread(&val_left, source, left.x * sizeof(float4), left.y, cudaBoundaryModeClamp);
		surf2Dread(&val_rigth, source, right.x * sizeof(float4), right.y, cudaBoundaryModeClamp);
		float2 vTop = make_float2(val_top.x * 2 - 1, val_top.y * 2 - 1);
		float2 vRight = make_float2(val_rigth.x * 2 - 1, val_rigth.y * 2 - 1);
		float2 vLeft = make_float2(val_left.x * 2 - 1, val_left.y * 2 - 1);
		float2 vBot = make_float2(val_bot.x * 2 - 1, val_bot.y * 2 - 1);

		float div = (vRight.x - vLeft.x + vBot.y - vTop.y) / 2.f;
		float4 center;
		surf2Dread(&center, source, x * sizeof(float4), y, cudaBoundaryModeClamp);
		surf2Dwrite(make_float4(center.x, center.y, center.z, div), target, x * sizeof(float4), y);
	}
}

__global__ void jacobyViscousDiffusion(
	cudaSurfaceObject_t source,
	cudaSurfaceObject_t target,
	unsigned width, unsigned height) 
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		// Getting 4 adiacent pixels positions
		int2 right = make_int2(x + 1, y);
		int2 left = make_int2(x - 1, y);
		int2 top = make_int2(x, y - 1);
		int2 bot = make_int2(x, y + 1);
		// Getting the 4 pixels values
		float4 val_top, val_bot, val_left, val_rigth , center;
		surf2Dread(&val_top, source, left.x * sizeof(float4), left.y, cudaBoundaryModeClamp);
		surf2Dread(&val_bot, source, bot.x * sizeof(float4), bot.y, cudaBoundaryModeClamp);
		surf2Dread(&val_left, source, top.x * sizeof(float4), top.y, cudaBoundaryModeClamp);
		surf2Dread(&val_rigth, source, right.x * sizeof(float4), right.y, cudaBoundaryModeClamp);
		surf2Dread(&center, source, x * sizeof(float4), y, cudaBoundaryModeClamp);
		float alpha = -1.f;
		// z value is the pressure, w value is the divergence
		center.z = (val_top.z + val_bot.z + val_left.z + val_rigth.z + alpha * center.w) / 4.f;
		surf2Dwrite(center, target, x * sizeof(float4), y);
	}
}

__global__ void applyGradient(
	cudaSurfaceObject_t source,
	cudaSurfaceObject_t target,
	unsigned width, unsigned height)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		// Getting 4 adiacent pixels positions
		int2 right = make_int2(x + 1, y);
		int2 left = make_int2(x - 1, y);
		int2 top = make_int2(x, y - 1);
		int2 bot = make_int2(x, y + 1);
		// Getting the 4 pixels values
		float4 val_top, val_bot, val_left, val_rigth, center;
		surf2Dread(&val_top, source, top.x * sizeof(float4), top.y, cudaBoundaryModeClamp);
		surf2Dread(&val_bot, source, bot.x * sizeof(float4), bot.y, cudaBoundaryModeClamp);
		surf2Dread(&val_left, source, left.x * sizeof(float4), left.y, cudaBoundaryModeClamp);
		surf2Dread(&val_rigth, source, right.x * sizeof(float4), right.y, cudaBoundaryModeClamp);
		surf2Dread(&center, source, x * sizeof(float4), y, cudaBoundaryModeClamp);
		// z is the pressure, x and y are the velodity components
		float2 gradient;
		gradient.x = 0.5 * (val_rigth.z - val_left.z);
		gradient.y = 0.5 * (val_bot.z - val_top.z);
		center.x = center.x - gradient.x;
		center.y = center.y - gradient.y;
		surf2Dwrite(center, target, x * sizeof(float4), y);
	}
}


__global__ void enforceBoundaryConditions(cudaSurfaceObject_t target, unsigned width, unsigned height)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	// If it's not the border exit, this is not an optimal solution but is simple to code...
	if ((x > 0 && x < width - 1 && y < height - 1 && y > 0) || (x >= width || y>=height)) { return; }
	int offset_x = 0, offset_y = 0;
	//Edges
	if (x == 0) { offset_x = 1; }
	else { offset_x = -1; }
	if (y == 0) { offset_y = 1; }
	else { offset_y = -1; }
	float4 val;
	surf2Dread(&val, target, (x+offset_x) * sizeof(float4), (y+offset_y));
	surf2Dwrite(make_float4(0.5f,0.5f,val.z,val.w), target, x * sizeof(float4), y);
}

__global__ void advectDyeConcentration(cudaSurfaceObject_t velocity, 
	cudaSurfaceObject_t dye_in, cudaSurfaceObject_t dye_out, SimulationData data)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= 0 && x < data.dye_width && y < data.dye_height && y >= 0) {
		float2 sim_coord = make_float2(x * (data.sim_width-1) / (float)data.dye_width, y * (data.sim_height-1) / (float)data.dye_height);
		float2 old_pos;
		//float4 local_color;
		float3 local_velocity = bilerp(velocity, sim_coord, data.sim_width, data.sim_height);
		// follow the velocity field "back in time"
		// dt is in milliseconds, color channels are shifted with *2-1 to indicate negative directions
		old_pos.x = (float)x - data.dt * ((local_velocity.x * 2 - 1) * data.dye_width / ((float)data.sim_width-1));
		old_pos.y = (float)y - data.dt * ((local_velocity.y * 2 - 1) * data.dye_height / ((float)data.sim_height-1));
		// interpolate and write to the output
		float3 sample = bilerp(dye_in, old_pos, data.dye_width, data.dye_height);
		float decay = 1.f + data.density_diffusion * data.dt;
		surf2Dwrite(make_float4(sample.x / decay, sample.y / decay, sample.z / decay, 1.0), dye_out, x * sizeof(float4), y);
	}	
}

__global__ void applyDye( cudaSurfaceObject_t dye_out,
	unsigned width, unsigned height, SplatData splat)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= 0 && x < width && y < height && y >= 0) 
	{
		float influence = 1.0f * exp(-((x - splat.splat_posx * width)*(x - splat.splat_posx * width) +
			(y - splat.splat_posy * height)*(y - splat.splat_posy * height)) /
			((splat.splat_radius*height)*(splat.splat_radius*height)*2) );
		influence = clampTo_0_1(influence);
		float4 current_val;
		surf2Dread(&current_val, dye_out, x * sizeof(float4), y, cudaSurfaceBoundaryMode::cudaBoundaryModeClamp);
		current_val.x += splat.splat_color[0] * influence;
		current_val.y += splat.splat_color[1] * influence;
		current_val.z += splat.splat_color[2] * influence;
		current_val.x = clampTo_0_1(current_val.x);
		current_val.y = clampTo_0_1(current_val.y);
		current_val.z = clampTo_0_1(current_val.z);
		//printf("current_val.x = %f, current_val.y = %f, influence = %f\n", current_val.x, current_val.y, influence);
		surf2Dwrite(current_val, dye_out, x * sizeof(float4), y);
	}
}

void launchSimKernels(SimulationData &data)
{
	dim3 blockSize(data.block_width, data.block_heigth, 1);
	dim3 simGridSize(data.sim_width / blockSize.x + 1, data.sim_height / blockSize.y + 1);
	dim3 dyeGridSize(data.dye_width / blockSize.x + 1, data.dye_height / blockSize.y + 1);
	//call CUDA kernel, writing results to texture through surface
	cudaSurfaceObject_t source, target, dye_in, dye_out;
	cudaResourceDesc description;
	memset(&description, 0, sizeof(description));
	description.resType = cudaResourceTypeArray;
	description.res.array.array = data.prev()->vector_field;
	checkCudaErrors(cudaCreateSurfaceObject(&source, &description));
	description.res.array.array = data.next()->vector_field;
	checkCudaErrors(cudaCreateSurfaceObject(&target, &description));
	description.res.array.array = data.prev()->dye_texture;
	checkCudaErrors(cudaCreateSurfaceObject(&dye_in, &description));
	description.res.array.array = data.next()->dye_texture;
	checkCudaErrors(cudaCreateSurfaceObject(&dye_out, &description));

	advectionStep << < simGridSize, blockSize >> > (source, target, data.sim_width, data.sim_height, data);
	if (data.splat.splatting) {
		forceApplicationStep << < simGridSize, blockSize >> > (target, data.sim_width, data.sim_height, data.splat);
		applyDye << < dyeGridSize, blockSize >> > (dye_in, data.dye_width, data.dye_height, data.splat);
	}

	divergence << < simGridSize, blockSize >> > (target, source, data.sim_width, data.sim_height);

	for (int i = 0; i < data.iterations / 2; i++) {	// 2 calls to invert the texture and cycle properly
		jacobyViscousDiffusion << < simGridSize, blockSize >> > (source, target, data.sim_width, data.sim_height);
		jacobyViscousDiffusion << < simGridSize, blockSize >> > (target, source, data.sim_width, data.sim_height);
	}
	enforceBoundaryConditions << < simGridSize, blockSize >> > (source, data.sim_width, data.sim_height);

	applyGradient << < simGridSize, blockSize >> > (source, target, data.sim_width, data.sim_height);
	// here we use a bigger GRID to advect the dye
	advectDyeConcentration << < dyeGridSize, blockSize >> > (target, dye_in, dye_out, data);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaDestroySurfaceObject(source));
	checkCudaErrors(cudaDestroySurfaceObject(target));
	checkCudaErrors(cudaDestroySurfaceObject(dye_in));
	checkCudaErrors(cudaDestroySurfaceObject(dye_out));
}



