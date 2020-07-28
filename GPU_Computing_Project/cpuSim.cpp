#include "simulation.h"

float _clampTo_0_1(float val) {
	if (val < 0.f) val = 0;
	if (val > 1.0f) val = 1.f;
	return val;
}

void MYsurf2Dread(float4 *out, std::vector<float> *pixels, int x, int y, unsigned w, unsigned h) 
{
	if (x > w - 1) { x = w - 1; }
	if (y > h - 1) { y = h - 1; }
	if (x < 0) { x = 0; }
	if (y < 0) { y = 0; }
	//printf("%d  %d\n", x, y);
	out->x = pixels->at(4 * w*y + 4*x);
	out->y = pixels->at(4 * w*y + 4*x + 1);
	out->z = pixels->at(4 * w*y + 4*x + 2);
	out->w = pixels->at(4 * w*y + 4*x + 3);
}

void MYsurf2Dwrite(float4 val,std::vector<float> *pixels, int x, int y, unsigned w, unsigned h) {
	if (x > w - 1) { x = w - 1; }
	if (y > h - 1) { y = h - 1; }
	if (x < 0) { x = 0; }
	if (y < 0) { y = 0; }
	pixels->data()[4 *w*y + 4*x ] = val.x;
	pixels->data()[4 *w*y + 4*x + 1] = val.y;
	pixels->data()[4 *w*y + 4*x + 2] = val.z;
	pixels->data()[4 *w*y + 4*x + 3] = val.w;
}


float3 bilerp(std::vector<float> *source, float2 coord, unsigned width, unsigned height)
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
	MYsurf2Dread(&color_br, source, bot_right.x, bot_right.y, width, height);
	MYsurf2Dread(&color_bl, source, bot_left.x, bot_left.y, width, height);
	MYsurf2Dread(&color_tl, source, top_left.x, top_left.y, width, height);
	MYsurf2Dread(&color_tr, source, top_right.x, top_right.y, width, height);

	//horizontal interpolation for the BOTTOM pixels
	float Xdiff_left = coord.x - (float)bot_left.x;
	float Xdiff_right = (float)bot_right.x - coord.x;
	float Hx1 = color_bl.x * Xdiff_right + color_br.x * Xdiff_left; // x comp
	float Hy1 = color_bl.y * Xdiff_right + color_br.y * Xdiff_left; // y comp
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

	return make_float3(X, Y, Z);
}

void advectionStep(
	std::vector<float> *source,
	std::vector<float> *target,
	unsigned width, unsigned height, SimulationData &data, unsigned x, unsigned y)
{
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		float2 old_pos;
		float4 local_val = make_float4(0, 0, 0, 0);
		MYsurf2Dread(&local_val, source, x, y, width, height);
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
		MYsurf2Dwrite(make_float4(newV.x, newV.y, local_val.z, 1.0), target, x, y, width, height);
	}
}

void forceApplicationStep(std::vector<float> *target, unsigned width, unsigned height, SplatData &splat, unsigned x, unsigned y)
{
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		float forceX = splat.splat_impuls_X * width;
		float forceY = splat.splat_impuls_Y * height;
		float influence = 1.0f * exp(-((x - splat.splat_posx * width)*(x - splat.splat_posx * width) +
			(y - splat.splat_posy * height)*(y - splat.splat_posy * height)) /
			((splat.splat_radius * height)*(splat.splat_radius * height) * 2));
		influence = _clampTo_0_1(influence);
		float4 current_val;
		MYsurf2Dread(&current_val, target, x, y, width, height);
		current_val.x += forceX * influence;
		current_val.y += forceY * influence;
		current_val.x = _clampTo_0_1(current_val.x);
		current_val.y = _clampTo_0_1(current_val.y);
		MYsurf2Dwrite(current_val, target, x , y, width, height);
	}
}

__global__ void divergence(std::vector<float> *source, std::vector<float> *target, unsigned width, unsigned height, unsigned x, unsigned y)
{
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		// Getting 4 adiacent pixels positions
		int2 right = make_int2(x + 1, y);
		int2 left = make_int2(x - 1, y);
		int2 top = make_int2(x, y - 1);
		int2 bot = make_int2(x, y + 1);
		// Getting the 4 pixels values
		float4 val_top, val_bot, val_left, val_rigth;
		MYsurf2Dread(&val_top, source, top.x , top.y, width, height);
		MYsurf2Dread(&val_bot, source, bot.x , bot.y, width, height);
		MYsurf2Dread(&val_left, source, left.x , left.y, width, height);
		MYsurf2Dread(&val_rigth, source, right.x , right.y, width, height);
		float2 vTop = make_float2(val_top.x * 2 - 1, val_top.y * 2 - 1);
		float2 vRight = make_float2(val_rigth.x * 2 - 1, val_rigth.y * 2 - 1);
		float2 vLeft = make_float2(val_left.x * 2 - 1, val_left.y * 2 - 1);
		float2 vBot = make_float2(val_bot.x * 2 - 1, val_bot.y * 2 - 1);

		float div = (vRight.x - vLeft.x + vBot.y - vTop.y) / 2.f;
		float4 center;
		MYsurf2Dread(&center, source, x, y, width, height);
		MYsurf2Dwrite(make_float4(center.x, center.y, center.z, div), target, x, y, width, height);
	}
}

__global__ void jacobyViscousDiffusion(
	std::vector<float> *source,
	std::vector<float> *target,
	unsigned width, unsigned height, unsigned x, unsigned y)
{
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		// Getting 4 adiacent pixels positions
		int2 right = make_int2(x + 1, y);
		int2 left = make_int2(x - 1, y);
		int2 top = make_int2(x, y - 1);
		int2 bot = make_int2(x, y + 1);
		// Getting the 4 pixels values
		float4 val_top, val_bot, val_left, val_rigth, center;
		MYsurf2Dread(&val_top, source, left.x, left.y, width, height);
		MYsurf2Dread(&val_bot, source, bot.x , bot.y, width, height);
		MYsurf2Dread(&val_left, source, top.x , top.y, width, height);
		MYsurf2Dread(&val_rigth, source, right.x, right.y, width, height);
		MYsurf2Dread(&center, source, x, y, width, height);
		float alpha = -1.f;
		// z value is the pressure, w value is the divergence
		center.z = (val_top.z + val_bot.z + val_left.z + val_rigth.z + alpha * center.w) / 4.f;
		MYsurf2Dwrite(center, target, x, y, width, height);
	}
}

__global__ void applyGradient(
	std::vector<float> *source,
	std::vector<float> *target,
	unsigned width, unsigned height, unsigned x, unsigned y)
{
	if (x > 0 && x < width - 1 && y < height - 1 && y > 0) {
		// Getting 4 adiacent pixels positions
		int2 right = make_int2(x + 1, y);
		int2 left = make_int2(x - 1, y);
		int2 top = make_int2(x, y - 1);
		int2 bot = make_int2(x, y + 1);
		// Getting the 4 pixels values
		float4 val_top, val_bot, val_left, val_rigth, center;
		MYsurf2Dread(&val_top, source, top.x, top.y, width, height);
		MYsurf2Dread(&val_bot, source, bot.x, bot.y, width, height);
		MYsurf2Dread(&val_left, source, left.x, left.y, width, height);
		MYsurf2Dread(&val_rigth, source, right.x, right.y, width, height);
		MYsurf2Dread(&center, source, x, y, width, height);
		// z is the pressure, x and y are the velodity components
		float2 gradient;
		gradient.x = 0.5 * (val_rigth.z - val_left.z);
		gradient.y = 0.5 * (val_bot.z - val_top.z);
		center.x = center.x - gradient.x;
		center.y = center.y - gradient.y;
		MYsurf2Dwrite(center, target, x, y, width, height);
	}
}


void enforceBoundaryConditions(std::vector<float> *target, unsigned width, unsigned height, unsigned x, unsigned y)
{
	// If it's not the border exit, this is not an optimal solution but is simple to code...
	if ((x > 0 && x < width - 1 && y < height - 1 && y > 0) || (x >= width || y >= height)) { return; }
	int offset_x = 0, offset_y = 0;
	//Edges
	if (x == 0) { offset_x = 1; }
	else { offset_x = -1; }
	if (y == 0) { offset_y = 1; }
	else { offset_y = -1; }
	float4 val;
	MYsurf2Dread(&val, target, (x + offset_x) , (y + offset_y), width, height);
	MYsurf2Dwrite(make_float4(0.5f, 0.5f, val.z, val.w), target, x, y, width, height);
}

void advectDyeConcentration(std::vector<float> *velocity, std::vector<float> *dye_in, 
	std::vector<float> *dye_out,unsigned x, unsigned y, SimulationData &data)
{
	if (x >= 0 && x < data.dye_width && y < data.dye_height && y >= 0) {
		float2 sim_coord = make_float2(x * (data.sim_width - 1) / (float)data.dye_width, y * (data.sim_height - 1) / (float)data.dye_height);
		float2 old_pos;
		float3 local_velocity = bilerp(velocity, sim_coord, data.sim_width, data.sim_height);
		// follow the velocity field "back in time"
		// dt is in milliseconds, color channels are shifted with *2-1 to indicate negative directions
		old_pos.x = (float)x - data.dt * ((local_velocity.x * 2 - 1) * data.dye_width / ((float)data.sim_width - 1));
		old_pos.y = (float)y - data.dt * ((local_velocity.y * 2 - 1) * data.dye_height / ((float)data.sim_height - 1));
		// interpolate and write to the output
		float3 sample = bilerp(dye_in, old_pos, data.dye_width, data.dye_height);
		float decay = 1.f + data.density_diffusion * data.dt;
		MYsurf2Dwrite(make_float4(sample.x / decay, sample.y / decay, sample.z / decay, 1.0), dye_out, x , y, data.dye_width, data.dye_height);
	}
}

void applyDye(std::vector<float> *dye_out,
	unsigned width, unsigned height, SplatData splat, unsigned x, unsigned y)
{
	if (x >= 0 && x < width && y < height && y >= 0)
	{
		float influence = 1.0f * exp(-((x - splat.splat_posx * width)*(x - splat.splat_posx * width) +
			(y - splat.splat_posy * height)*(y - splat.splat_posy * height)) /
			((splat.splat_radius*height)*(splat.splat_radius*height) * 2));
		influence = _clampTo_0_1(influence);
		float4 current_val;
		MYsurf2Dread(&current_val, dye_out, x, y, width, height);
		current_val.x += splat.splat_color[0] * influence;
		current_val.y += splat.splat_color[1] * influence;
		current_val.z += splat.splat_color[2] * influence;
		current_val.x = _clampTo_0_1(current_val.x);
		current_val.y = _clampTo_0_1(current_val.y);
		current_val.z = _clampTo_0_1(current_val.z);
		//printf("current_val.x = %f, current_val.y = %f, influence = %f\n", current_val.x, current_val.y, influence);
		MYsurf2Dwrite(current_val, dye_out, x , y, width, height);
	}
}



void launchCPUImpl(SimulationData &data) {

	std::vector<float> * source = data.prev()->cpu_vector_field;
	std::vector<float> * target = data.next()->cpu_vector_field;

	std::vector<float> * dye_in = data.prev()->cpu_color_field;
	std::vector<float> * dye_out = data.next()->cpu_color_field;

	for (int r = 0; r < data.sim_height; r++) {
		for (int c = 0; c < data.sim_width; c++) {
			advectionStep(source, target, data.sim_width, data.sim_height, data,c,r);
		}
	}
	if (data.splat.splatting) {
		for (int r = 0; r < data.sim_height; r++) {
			for (int c = 0; c < data.sim_width; c++) {
				forceApplicationStep(target, data.sim_width, data.sim_height, data.splat,c,r);
			}
		}
		for (int r = 0; r < data.dye_height; r++) {
			for (int c = 0; c < data.dye_width; c++) {
				applyDye(dye_in, data.dye_width, data.dye_height, data.splat,c,r);
			}
		}
	}
	for (int r = 0; r < data.sim_height; r++) {
		for (int c = 0; c < data.sim_width; c++) {
			divergence(target, source, data.sim_width, data.sim_height,c,r);
		}
	}

	for (int i = 0; i < data.iterations / 2; i++) {	// 2 calls to invert the texture and cycle properly
		for (int r = 0; r < data.sim_height; r++) {
			for (int c = 0; c < data.sim_width; c++) {
				jacobyViscousDiffusion(source, target, data.sim_width, data.sim_height,c,r);
			}
		}
		for (int r = 0; r < data.sim_height; r++) {
			for (int c = 0; c < data.sim_width; c++) {
				jacobyViscousDiffusion(target, source, data.sim_width, data.sim_height,c,r);
			}
		}
	}
	for (int r = 0; r < data.sim_height; r++) {
		for (int c = 0; c < data.sim_width; c++) {
			enforceBoundaryConditions(source, data.sim_width, data.sim_height,c,r);
		}
	}
	for (int r = 0; r < data.sim_height; r++) {
		for (int c = 0; c < data.sim_width; c++) {
			applyGradient(source, target, data.sim_width, data.sim_height,c,r);
		}
	}
	for (int r = 0; r < data.dye_height; r++) {
		for (int c = 0; c < data.dye_width; c++) {
			advectDyeConcentration(target, dye_in, dye_out, c, r, data);
		}
	}
}