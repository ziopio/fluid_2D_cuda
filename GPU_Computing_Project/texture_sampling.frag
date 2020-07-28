#version 330 core

in vec2 UV;

out vec4 frag_color;

uniform bool debug_flag;
uniform sampler2D velocity_sampler;
uniform sampler2D color_sampler;

void main(void)
{
	if(debug_flag){
		frag_color = texture(velocity_sampler, UV);
	} else {
		frag_color = texture(color_sampler, UV);
	}
}
