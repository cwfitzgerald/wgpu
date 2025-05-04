#version 310 es

precision highp float;
precision highp int;

uniform uint naga_vs_first_instance;

uniform uint naga_vs_first_vertex;

struct gen_gl_PerVertex {
    vec4 gen_gl_Position;
    float gen_gl_PointSize;
    float gen_gl_ClipDistance[1];
    float gen_gl_CullDistance[1];
};
gen_gl_PerVertex unnamed = gen_gl_PerVertex(vec4(0.0, 0.0, 0.0, 1.0), 1.0, float[1](0.0), float[1](0.0));

int gen_gl_BaseVertex_1 = 0;

int gen_gl_BaseInstance_1 = 0;


void main_1() {
    int _e5 = gen_gl_BaseVertex_1;
    int _e7 = gen_gl_BaseInstance_1;
    unnamed.gen_gl_Position = vec4(float(_e5), float(_e7), 0.0, 1.0);
    return;
}

void main() {
    uint gen_gl_BaseVertex = naga_vs_first_vertex;
    uint gen_gl_BaseInstance = naga_vs_first_instance;
    gen_gl_BaseVertex_1 = int(gen_gl_BaseVertex);
    gen_gl_BaseInstance_1 = int(gen_gl_BaseInstance);
    main_1();
    vec4 _e8 = unnamed.gen_gl_Position;
    gl_Position = _e8;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

