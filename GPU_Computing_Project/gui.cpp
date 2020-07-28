#include "gui.h"

//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"

#include "..\imgui\imgui.h"
#include "..\imgui\imgui_impl_glfw.h"
#include "..\imgui\imgui_impl_opengl3.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include <GL/gl3w.h> // Initialize with gl3wInit()


// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

namespace gui {
	// Our state
	bool stop_time = false; // ZA-WARUDO!!!
	bool debug_flag = false;
	bool use_CPU = false;
	bool reload_sim_buffers;
	bool reload_dye_buffers;
	bool cuda_supported;
	std::string gpu_name = "no-gpu";
	unsigned w_width, w_height;
	static const char* available_resolutions[] = { "32","64","128","256","512","1024","2048" };
	static std::chrono::nanoseconds delta;
	static std::chrono::time_point<std::chrono::steady_clock> last_time;
	static GLFWwindow* window;
	static bool show_demo_window = false;
	static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	static void (*mouseButtonCallback)(int,int,int);
	static void(*cursorPosCallback)(double, double);

	static void glfw_error_callback(int error, const char* description)
	{
		fprintf(stderr, "Glfw Error %d: %s\n", error, description);
	}

	void framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		w_width = width;
		w_height = height;
	}

	void handle_position(GLFWwindow* window, double xpos, double ypos)
	{
		ImGuiIO& io = ImGui::GetIO();
		if (!io.WantCaptureMouse) {
			cursorPosCallback(xpos, ypos);
		}
	}

	void handle_mouse_click(GLFWwindow* window, int button, int action, int mods)
	{
		ImGuiIO& io = ImGui::GetIO();
		if (!io.WantCaptureMouse) {
			mouseButtonCallback(button, action, mods);
		}
	}

//  Helper function to load vertex and fragment shader files
// Create a NULL-terminated string by reading the provided file
	static char* readShaderSource(const char* shaderFile)
	{
		FILE* fp = fopen(shaderFile, "rb");

		if (fp == NULL) { return NULL; }

		fseek(fp, 0L, SEEK_END);
		long size = ftell(fp);

		fseek(fp, 0L, SEEK_SET);
		char* buf = new char[size + 1];
		fread(buf, 1, size, fp);

		buf[size] = '\0';
		fclose(fp);

		return buf;
	}

	// Create a GLSL program object from vertex and fragment shader files
	unsigned initShader(const char * vShaderFile, const char * fShaderFile)
	{
		// vertex shader
		int vertexShader = glCreateShader(GL_VERTEX_SHADER);
		const char * source = readShaderSource(vShaderFile);
		glShaderSource(vertexShader, 1, &source, NULL);
		glCompileShader(vertexShader);
		free((void*)source);
		// check for shader compile errors
		int success;
		char infoLog[512];
		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// fragment shader
		int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		source = readShaderSource(fShaderFile);
		glShaderSource(fragmentShader, 1, &source, NULL);
		glCompileShader(fragmentShader);
		free((void*)source);

		// check for shader compile errors
		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// link shaders
		int shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);
		// check for linking errors
		glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
		return shaderProgram;
	}

	unsigned allocateOpenGLTexture(unsigned width, unsigned height, unsigned init_value) {
		//int w, h, c;
		//auto pixels = stbi_load("texture.png",&w,&h,&c, STBI_rgb_alpha);
		std::vector<GLubyte> pixels(width * height * 4, init_value);
		GLuint gl_texture;
		glGenTextures(1, &gl_texture);
		glBindTexture(GL_TEXTURE_2D, gl_texture);
		// set basic parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE
		);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// Create texture data from 4-component unsigned byte to 4 floats
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
		glBindTexture(GL_TEXTURE_2D, 0);
		//stbi_image_free(pixels);
		return gl_texture;
	}

	void loadDataToTexture(unsigned textureID, float* data, unsigned w, unsigned h)
	{
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, data);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void initGui(unsigned width, unsigned height, void(*mouseButtonCallback)(int, int, int), void(*cursoPosCallback)(double, double))
	{
		w_width = width;
		w_height = height;
		gui::last_time = std::chrono::high_resolution_clock::now();
		// Setup window
		glfwSetErrorCallback(glfw_error_callback);
		if (!glfwInit()) {
			std::cout << "GLFW failed initialization..." << std::endl;
			exit(1);
		}

		const char* glsl_version = "#version 330";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);

		// Create window with graphics context
		window = glfwCreateWindow(width, height, "CUDA Stable-Fluids Simulation", NULL, NULL);
		if (window == NULL) {
			std::cout << "GLFW failed windows creation..." << std::endl;
			exit(1);
		}
		glfwMakeContextCurrent(window);
		glfwSwapInterval(0); // Disable vsync

		// Initialize OpenGL loader
		bool err = gl3wInit() != 0;
		if (err)
		{
			std::cout << "Failed to initialize OpenGL loader!" << std::endl;
			exit(1);
		}

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();

		//Custom callbacks
		gui::mouseButtonCallback = mouseButtonCallback;
		gui::cursorPosCallback = cursoPosCallback;
		glfwSetMouseButtonCallback(window, gui::handle_mouse_click);
		glfwSetCursorPosCallback(window, gui::handle_position);
		glfwSetFramebufferSizeCallback(window, gui::framebuffer_size_callback);

		// Setup Platform/Renderer bindings
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);

	}

	bool shouldCloseApp()
	{
		return glfwWindowShouldClose(window);
	}

	void preRender()
	{
		auto now = std::chrono::high_resolution_clock::now();
		delta = now - last_time;
		last_time = now;
		glfwPollEvents();
		//int display_w, display_h;
		//glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, w_width, w_height);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	double getDeltaTime()
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(delta).count() / 1000.0;
	}

	void putCurrentResIndex(SimulationData &sim, int&selectedSimRes, int&selectedDyeRes) {
		// not too proud of this... :/
		for (int n = 0; n < IM_ARRAYSIZE(available_resolutions); n++)
		{
			if (!strcmp(available_resolutions[n], (std::stringstream() << sim.sim_width).str().c_str())) {
				selectedSimRes = n;
			}
			if (!strcmp(available_resolutions[n], (std::stringstream() << sim.dye_width).str().c_str())) {
				selectedDyeRes = n;
			}
		}
	}

	float * computeRainbow() {
		static float rgb[3] = { 1, 0, 0 };
		static int phase = 0, counter = 0;
		const float step = 0.01;

		switch (phase) {
		case 0: rgb[1] += step;
			break;
		case 1: rgb[0] -= step;
			break;
		case 2: rgb[2] += step;
			break;
		case 3: rgb[1] -= step;
			break;
		case 4: rgb[0] += step;
			break;
		case 5: rgb[2] -= step;
			break;
		default:
			break;
		}
		counter++;
		if (counter >= 100) {
			counter = 0;
			phase < 5 ? phase++ : phase = 0;
		}
		return rgb;
	}

	void renderGUIOverlay(SimulationData &sim)
	{
		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		if (show_demo_window) {
			ImGui::ShowDemoWindow(&show_demo_window);
		}
		{
			ImGui::SetNextWindowPos(ImVec2(w_width - 350,0));
			ImGui::Begin("Configuration",nullptr,ImGuiWindowFlags_AlwaysAutoResize);  
			ImGui::Text("Fast Fluid Dynamics Simulation");
			ImGui::SameLine(); ImGui::TextDisabled("(?)");
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
				ImGui::TextUnformatted(
"What you see is a basic implementation of the method \"Stable Fluids\", \
proposed by Jos Stam (1999) which is then based on the Navier-Stokes equations for fluids. \
This work follows an article took from the book \"GPU gems\".");
				ImGui::PopTextWrapPos();
				ImGui::EndTooltip();
			}

			if (cuda_supported) {
				static int device = 0;
				//ImGui::Text("Try to switch from parallel to sequential");
				ImGui::RadioButton(gui::gpu_name.c_str(), &device, 0); ImGui::SameLine();
				ImGui::RadioButton("CPU (single-thread)", &device, 1);
				gui::use_CPU = device;
				if (ImGui::CollapsingHeader("Cuda")) {

					ImGui::SliderInt("Block Width", &sim.block_width, 1, 32);
					ImGui::SliderInt("Block Height", &sim.block_heigth, 1, 32);
				}
			}

			if (ImGui::CollapsingHeader("Simulation")) {
				static int selectedSimRes = -1, selectedDyeRes = -1;
				if (selectedSimRes == -1 || selectedDyeRes == -1) putCurrentResIndex(sim,selectedSimRes, selectedDyeRes);
				if (ImGui::Combo("Sim resolution", &selectedSimRes, available_resolutions, IM_ARRAYSIZE(available_resolutions))) {
					sim.sim_height = sim.sim_width = atoi(available_resolutions[selectedSimRes]);
					reload_sim_buffers = true;
				}
				if (ImGui::Combo("Dye resolution", &selectedDyeRes, available_resolutions, IM_ARRAYSIZE(available_resolutions))) {
					sim.dye_height = sim.dye_width = atoi(available_resolutions[selectedDyeRes]);
					reload_dye_buffers = true;
				}

				ImGui::SliderInt("Jacobi Iterations", &sim.iterations, 2, 100);
				ImGui::SliderFloat("Velocity diffusion", &sim.viscosity, 0.f,0.02f);
				ImGui::SliderFloat("Dye Diffusion", &sim.density_diffusion, 0.f, 0.02f);
			}
			if (ImGui::CollapsingHeader("Splat")) {
				ImGui::SliderFloat("Splat Radius", &sim.splat.splat_radius, 0.005f, 0.05f);
				ImGui::ColorPicker3("Splat color", sim.splat.splat_color,
					ImGuiColorEditFlags_::ImGuiColorEditFlags_NoSidePreview |
					ImGuiColorEditFlags_::ImGuiColorEditFlags_PickerHueWheel |
					ImGuiColorEditFlags_::ImGuiColorEditFlags_NoInputs |
					ImGuiColorEditFlags_::ImGuiColorEditFlags_NoAlpha); 
				//ImGui::SameLine();
				static bool rainbow;
				ImGui::Checkbox("Rainbow!! :P", &rainbow);
				if (rainbow) {
					float * color = computeRainbow();
					sim.splat.splat_color[0] = color[0];
					sim.splat.splat_color[1] = color[1];
					sim.splat.splat_color[2] = color[2];
				}
			}
			static bool vsync = true;
			ImGui::Checkbox("Enable V-sync", &vsync); ImGui::SameLine(); 
			ImGui::Checkbox("Debug", &gui::debug_flag); ImGui::SameLine();
			ImGui::Checkbox("Stop Fluid", &stop_time);
			glfwSwapInterval(vsync);


			ImGui::Checkbox("Demo window", &show_demo_window);
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", gui::getDeltaTime(), ImGui::GetIO().Framerate);
			ImGui::End();
		}

		// Rendering
		ImGui::Render();

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	void presentToScreen()
	{
		glfwSwapBuffers(window);
	}

	void cleanUp()
	{
		// Cleanup
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		glfwDestroyWindow(window);
		glfwTerminate();
	}
}
