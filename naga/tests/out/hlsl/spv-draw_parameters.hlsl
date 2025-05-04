struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0);

struct gl_PerVertex {
    float4 gl_Position : SV_Position;
    float gl_PointSize;
    float gl_ClipDistance[1];
    float gl_CullDistance[1];
    int _end_pad_0;
};

gl_PerVertex Constructgl_PerVertex(float4 arg0, float arg1, float arg2[1], float arg3[1]) {
    gl_PerVertex ret = (gl_PerVertex)0;
    ret.gl_Position = arg0;
    ret.gl_PointSize = arg1;
    ret.gl_ClipDistance = arg2;
    ret.gl_CullDistance = arg3;
    return ret;
}

typedef float ret_ZeroValuearray1_float_[1];
ret_ZeroValuearray1_float_ ZeroValuearray1_float_() {
    return (float[1])0;
}

static gl_PerVertex unnamed = Constructgl_PerVertex(float4(0.0, 0.0, 0.0, 1.0), 1.0, ZeroValuearray1_float_(), ZeroValuearray1_float_());
static int gl_BaseVertex_1 = (int)0;
static int gl_BaseInstance_1 = (int)0;

struct VertexInput_main {
};

void main_1()
{
    int _e5 = gl_BaseVertex_1;
    int _e7 = gl_BaseInstance_1;
    unnamed.gl_Position = float4(float(_e5), float(_e7), 0.0, 1.0);
    return;
}

float4 main(VertexInput_main vertexinput_main) : SV_Position
{
    uint gl_BaseVertex = _NagaConstants.first_vertex;
    uint gl_BaseInstance = _NagaConstants.first_instance;
    gl_BaseVertex_1 = int(gl_BaseVertex);
    gl_BaseInstance_1 = int(gl_BaseInstance);
    main_1();
    float4 _e8 = unnamed.gl_Position;
    return _e8;
}
