#version 460

void main() {
    // gl_DrawID is not supported.
    gl_Position = vec4(float(gl_BaseVertex), float(gl_BaseInstance), 0.0, 1.0);
}