#version 330 core

out vec2 UV;

void main()
{
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    UV.x = (x+1.0)*0.5;
    UV.y = (-y+1.0)*0.5;
    gl_Position = vec4(x, y, 1.0f, 1.0f);
}

// {
//     float x = -1.0 + float((gl_VertexID & 1) << 2);
//     float y = -1.0 + float((gl_VertexID & 2) << 1);
//     UV.x = (x+1.0)*0.5;
//     UV.y = (y+1.0)*0.5;
//     gl_Position = vec4(x, y, 1.0f, 1.0f);
// }
// void main()
// {
// 	UV = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
// 	gl_Position = vec4(UV * 2.0f - 1.0f, 0.0f, 1.0f);
// }
