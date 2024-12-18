// When compiling with FXC without strict mode, these keywords are actually case insensitive.
// If you compile with strict mode and specify a different casing like "Pass" instead in an identifier, FXC will give this error:
// "error X3086: alternate cases for 'pass' are deprecated in strict mode"
// This behavior is not documented anywhere, but as far as I can tell this is the full list.
pub const RESERVED_CASE_INSENSITIVE: &[&str] = &[
    "asm",
    "decl",
    "pass",
    "technique",
    "Texture1D",
    "Texture2D",
    "Texture3D",
    "TextureCube",
];

pub const RESERVED: &[&str] = &[
    // FXC keywords, from https://github.com/MicrosoftDocs/win32/blob/c885cb0c63b0e9be80c6a0e6512473ac6f4e771e/desktop-src/direct3dhlsl/dx-graphics-hlsl-appendix-keywords.md?plain=1#L99-L118
    "AppendStructuredBuffer",
    "asm",
    "asm_fragment",
    "BlendState",
    "bool",
    "break",
    "Buffer",
    "ByteAddressBuffer",
    "case",
    "cbuffer",
    "centroid",
    "class",
    "column_major",
    "compile",
    "compile_fragment",
    "CompileShader",
    "const",
    "continue",
    "ComputeShader",
    "ConsumeStructuredBuffer",
    "default",
    "DepthStencilState",
    "DepthStencilView",
    "discard",
    "do",
    "double",
    "DomainShader",
    "dword",
    "else",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "fxgroup",
    "GeometryShader",
    "groupshared",
    "half",
    "Hullshader",
    "if",
    "in",
    "inline",
    "inout",
    "InputPatch",
    "int",
    "interface",
    "line",
    "lineadj",
    "linear",
    "LineStream",
    "matrix",
    "min16float",
    "min10float",
    "min16int",
    "min12int",
    "min16uint",
    "namespace",
    "nointerpolation",
    "noperspective",
    "NULL",
    "out",
    "OutputPatch",
    "packoffset",
    "pass",
    "pixelfragment",
    "PixelShader",
    "point",
    "PointStream",
    "precise",
    "RasterizerState",
    "RenderTargetView",
    "return",
    "register",
    "row_major",
    "RWBuffer",
    "RWByteAddressBuffer",
    "RWStructuredBuffer",
    "RWTexture1D",
    "RWTexture1DArray",
    "RWTexture2D",
    "RWTexture2DArray",
    "RWTexture3D",
    "sample",
    "sampler",
    "SamplerState",
    "SamplerComparisonState",
    "shared",
    "snorm",
    "stateblock",
    "stateblock_state",
    "static",
    "string",
    "struct",
    "switch",
    "StructuredBuffer",
    "tbuffer",
    "technique",
    "technique10",
    "technique11",
    "texture",
    "Texture1D",
    "Texture1DArray",
    "Texture2D",
    "Texture2DArray",
    "Texture2DMS",
    "Texture2DMSArray",
    "Texture3D",
    "TextureCube",
    "TextureCubeArray",
    "true",
    "typedef",
    "triangle",
    "triangleadj",
    "TriangleStream",
    "uint",
    "uniform",
    "unorm",
    "unsigned",
    "vector",
    "vertexfragment",
    "VertexShader",
    "void",
    "volatile",
    "while",
    // FXC reserved keywords, from https://github.com/MicrosoftDocs/win32/blob/c885cb0c63b0e9be80c6a0e6512473ac6f4e771e/desktop-src/direct3dhlsl/dx-graphics-hlsl-appendix-reserved-words.md?plain=1#L19-L38
    "auto",
    "case",
    "catch",
    "char",
    "class",
    "const_cast",
    "default",
    "delete",
    "dynamic_cast",
    "enum",
    "explicit",
    "friend",
    "goto",
    "long",
    "mutable",
    "new",
    "operator",
    "private",
    "protected",
    "public",
    "reinterpret_cast",
    "short",
    "signed",
    "sizeof",
    "static_cast",
    "template",
    "this",
    "throw",
    "try",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    // FXC intrinsics, from https://github.com/MicrosoftDocs/win32/blob/1682b99e203708f6f5eda972d966e30f3c1588de/desktop-src/direct3dhlsl/dx-graphics-hlsl-intrinsic-functions.md?plain=1#L26-L165
    "abort",
    "abs",
    "acos",
    "all",
    "AllMemoryBarrier",
    "AllMemoryBarrierWithGroupSync",
    "any",
    "asdouble",
    "asfloat",
    "asin",
    "asint",
    "asuint",
    "atan",
    "atan2",
    "ceil",
    "CheckAccessFullyMapped",
    "clamp",
    "clip",
    "cos",
    "cosh",
    "countbits",
    "cross",
    "D3DCOLORtoUBYTE4",
    "ddx",
    "ddx_coarse",
    "ddx_fine",
    "ddy",
    "ddy_coarse",
    "ddy_fine",
    "degrees",
    "determinant",
    "DeviceMemoryBarrier",
    "DeviceMemoryBarrierWithGroupSync",
    "distance",
    "dot",
    "dst",
    "errorf",
    "EvaluateAttributeCentroid",
    "EvaluateAttributeAtSample",
    "EvaluateAttributeSnapped",
    "exp",
    "exp2",
    "f16tof32",
    "f32tof16",
    "faceforward",
    "firstbithigh",
    "firstbitlow",
    "floor",
    "fma",
    "fmod",
    "frac",
    "frexp",
    "fwidth",
    "GetRenderTargetSampleCount",
    "GetRenderTargetSamplePosition",
    "GroupMemoryBarrier",
    "GroupMemoryBarrierWithGroupSync",
    "InterlockedAdd",
    "InterlockedAnd",
    "InterlockedCompareExchange",
    "InterlockedCompareStore",
    "InterlockedExchange",
    "InterlockedMax",
    "InterlockedMin",
    "InterlockedOr",
    "InterlockedXor",
    "isfinite",
    "isinf",
    "isnan",
    "ldexp",
    "length",
    "lerp",
    "lit",
    "log",
    "log10",
    "log2",
    "mad",
    "max",
    "min",
    "modf",
    "msad4",
    "mul",
    "noise",
    "normalize",
    "pow",
    "printf",
    "Process2DQuadTessFactorsAvg",
    "Process2DQuadTessFactorsMax",
    "Process2DQuadTessFactorsMin",
    "ProcessIsolineTessFactors",
    "ProcessQuadTessFactorsAvg",
    "ProcessQuadTessFactorsMax",
    "ProcessQuadTessFactorsMin",
    "ProcessTriTessFactorsAvg",
    "ProcessTriTessFactorsMax",
    "ProcessTriTessFactorsMin",
    "radians",
    "rcp",
    "reflect",
    "refract",
    "reversebits",
    "round",
    "rsqrt",
    "saturate",
    "sign",
    "sin",
    "sincos",
    "sinh",
    "smoothstep",
    "sqrt",
    "step",
    "tan",
    "tanh",
    "tex1D",
    "tex1Dbias",
    "tex1Dgrad",
    "tex1Dlod",
    "tex1Dproj",
    "tex2D",
    "tex2Dbias",
    "tex2Dgrad",
    "tex2Dlod",
    "tex2Dproj",
    "tex3D",
    "tex3Dbias",
    "tex3Dgrad",
    "tex3Dlod",
    "tex3Dproj",
    "texCUBE",
    "texCUBEbias",
    "texCUBEgrad",
    "texCUBElod",
    "texCUBEproj",
    "transpose",
    "trunc",
    // DXC (reserved) keywords, from https://github.com/microsoft/DirectXShaderCompiler/blob/d5d478470d3020a438d3cb810b8d3fe0992e6709/tools/clang/include/clang/Basic/TokenKinds.def#L222-L648
    // with the KEYALL, KEYCXX, BOOLSUPPORT, WCHARSUPPORT, KEYHLSL options enabled (see https://github.com/microsoft/DirectXShaderCompiler/blob/d5d478470d3020a438d3cb810b8d3fe0992e6709/tools/clang/lib/Frontend/CompilerInvocation.cpp#L1199)
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_Complex",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local",
    "__func__",
    "__objc_yes",
    "__objc_no",
    "asm",
    "bool",
    "catch",
    "class",
    "const_cast",
    "delete",
    "dynamic_cast",
    "explicit",
    "export",
    "false",
    "friend",
    "mutable",
    "namespace",
    "new",
    "operator",
    "private",
    "protected",
    "public",
    "reinterpret_cast",
    "static_cast",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typename",
    "typeid",
    "using",
    "virtual",
    "wchar_t",
    "_Decimal32",
    "_Decimal64",
    "_Decimal128",
    "__null",
    "__alignof",
    "__attribute",
    "__builtin_choose_expr",
    "__builtin_offsetof",
    "__builtin_va_arg",
    "__extension__",
    "__imag",
    "__int128",
    "__label__",
    "__real",
    "__thread",
    "__FUNCTION__",
    "__PRETTY_FUNCTION__",
    "__is_nothrow_assignable",
    "__is_constructible",
    "__is_nothrow_constructible",
    "__has_nothrow_assign",
    "__has_nothrow_move_assign",
    "__has_nothrow_copy",
    "__has_nothrow_constructor",
    "__has_trivial_assign",
    "__has_trivial_move_assign",
    "__has_trivial_copy",
    "__has_trivial_constructor",
    "__has_trivial_move_constructor",
    "__has_trivial_destructor",
    "__has_virtual_destructor",
    "__is_abstract",
    "__is_base_of",
    "__is_class",
    "__is_convertible_to",
    "__is_empty",
    "__is_enum",
    "__is_final",
    "__is_literal",
    "__is_literal_type",
    "__is_pod",
    "__is_polymorphic",
    "__is_trivial",
    "__is_union",
    "__is_trivially_constructible",
    "__is_trivially_copyable",
    "__is_trivially_assignable",
    "__underlying_type",
    "__is_lvalue_expr",
    "__is_rvalue_expr",
    "__is_arithmetic",
    "__is_floating_point",
    "__is_integral",
    "__is_complete_type",
    "__is_void",
    "__is_array",
    "__is_function",
    "__is_reference",
    "__is_lvalue_reference",
    "__is_rvalue_reference",
    "__is_fundamental",
    "__is_object",
    "__is_scalar",
    "__is_compound",
    "__is_pointer",
    "__is_member_object_pointer",
    "__is_member_function_pointer",
    "__is_member_pointer",
    "__is_const",
    "__is_volatile",
    "__is_standard_layout",
    "__is_signed",
    "__is_unsigned",
    "__is_same",
    "__is_convertible",
    "__array_rank",
    "__array_extent",
    "__private_extern__",
    "__module_private__",
    "__declspec",
    "__cdecl",
    "__stdcall",
    "__fastcall",
    "__thiscall",
    "__vectorcall",
    "cbuffer",
    "tbuffer",
    "packoffset",
    "linear",
    "centroid",
    "nointerpolation",
    "noperspective",
    "sample",
    "column_major",
    "row_major",
    "in",
    "out",
    "inout",
    "uniform",
    "precise",
    "center",
    "shared",
    "groupshared",
    "discard",
    "snorm",
    "unorm",
    "point",
    "line",
    "lineadj",
    "triangle",
    "triangleadj",
    "globallycoherent",
    "interface",
    "sampler_state",
    "technique",
    "indices",
    "vertices",
    "primitives",
    "payload",
    "Technique",
    "technique10",
    "technique11",
    "__builtin_omp_required_simd_align",
    "__pascal",
    "__fp16",
    "__alignof__",
    "__asm",
    "__asm__",
    "__attribute__",
    "__complex",
    "__complex__",
    "__const",
    "__const__",
    "__decltype",
    "__imag__",
    "__inline",
    "__inline__",
    "__nullptr",
    "__real__",
    "__restrict",
    "__restrict__",
    "__signed",
    "__signed__",
    "__typeof",
    "__typeof__",
    "__volatile",
    "__volatile__",
    "_Nonnull",
    "_Nullable",
    "_Null_unspecified",
    "__builtin_convertvector",
    "__char16_t",
    "__char32_t",
    // DXC intrinsics, from https://github.com/microsoft/DirectXShaderCompiler/blob/18c9e114f9c314f93e68fbc72ce207d4ed2e65ae/utils/hct/gen_intrin_main.txt#L86-L376
    "D3DCOLORtoUBYTE4",
    "GetRenderTargetSampleCount",
    "GetRenderTargetSamplePosition",
    "abort",
    "abs",
    "acos",
    "all",
    "AllMemoryBarrier",
    "AllMemoryBarrierWithGroupSync",
    "any",
    "asdouble",
    "asfloat",
    "asfloat16",
    "asint16",
    "asin",
    "asint",
    "asuint",
    "asuint16",
    "atan",
    "atan2",
    "ceil",
    "clamp",
    "clip",
    "cos",
    "cosh",
    "countbits",
    "cross",
    "ddx",
    "ddx_coarse",
    "ddx_fine",
    "ddy",
    "ddy_coarse",
    "ddy_fine",
    "degrees",
    "determinant",
    "DeviceMemoryBarrier",
    "DeviceMemoryBarrierWithGroupSync",
    "distance",
    "dot",
    "dst",
    "EvaluateAttributeAtSample",
    "EvaluateAttributeCentroid",
    "EvaluateAttributeSnapped",
    "GetAttributeAtVertex",
    "exp",
    "exp2",
    "f16tof32",
    "f32tof16",
    "faceforward",
    "firstbithigh",
    "firstbitlow",
    "floor",
    "fma",
    "fmod",
    "frac",
    "frexp",
    "fwidth",
    "GroupMemoryBarrier",
    "GroupMemoryBarrierWithGroupSync",
    "InterlockedAdd",
    "InterlockedMin",
    "InterlockedMax",
    "InterlockedAnd",
    "InterlockedOr",
    "InterlockedXor",
    "InterlockedCompareStore",
    "InterlockedExchange",
    "InterlockedCompareExchange",
    "InterlockedCompareStoreFloatBitwise",
    "InterlockedCompareExchangeFloatBitwise",
    "isfinite",
    "isinf",
    "isnan",
    "ldexp",
    "length",
    "lerp",
    "lit",
    "log",
    "log10",
    "log2",
    "mad",
    "max",
    "min",
    "modf",
    "msad4",
    "mul",
    "normalize",
    "pow",
    "printf",
    "Process2DQuadTessFactorsAvg",
    "Process2DQuadTessFactorsMax",
    "Process2DQuadTessFactorsMin",
    "ProcessIsolineTessFactors",
    "ProcessQuadTessFactorsAvg",
    "ProcessQuadTessFactorsMax",
    "ProcessQuadTessFactorsMin",
    "ProcessTriTessFactorsAvg",
    "ProcessTriTessFactorsMax",
    "ProcessTriTessFactorsMin",
    "radians",
    "rcp",
    "reflect",
    "refract",
    "reversebits",
    "round",
    "rsqrt",
    "saturate",
    "sign",
    "sin",
    "sincos",
    "sinh",
    "smoothstep",
    "source_mark",
    "sqrt",
    "step",
    "tan",
    "tanh",
    "tex1D",
    "tex1Dbias",
    "tex1Dgrad",
    "tex1Dlod",
    "tex1Dproj",
    "tex2D",
    "tex2Dbias",
    "tex2Dgrad",
    "tex2Dlod",
    "tex2Dproj",
    "tex3D",
    "tex3Dbias",
    "tex3Dgrad",
    "tex3Dlod",
    "tex3Dproj",
    "texCUBE",
    "texCUBEbias",
    "texCUBEgrad",
    "texCUBElod",
    "texCUBEproj",
    "transpose",
    "trunc",
    "CheckAccessFullyMapped",
    "AddUint64",
    "NonUniformResourceIndex",
    "WaveIsFirstLane",
    "WaveGetLaneIndex",
    "WaveGetLaneCount",
    "WaveActiveAnyTrue",
    "WaveActiveAllTrue",
    "WaveActiveAllEqual",
    "WaveActiveBallot",
    "WaveReadLaneAt",
    "WaveReadLaneFirst",
    "WaveActiveCountBits",
    "WaveActiveSum",
    "WaveActiveProduct",
    "WaveActiveBitAnd",
    "WaveActiveBitOr",
    "WaveActiveBitXor",
    "WaveActiveMin",
    "WaveActiveMax",
    "WavePrefixCountBits",
    "WavePrefixSum",
    "WavePrefixProduct",
    "WaveMatch",
    "WaveMultiPrefixBitAnd",
    "WaveMultiPrefixBitOr",
    "WaveMultiPrefixBitXor",
    "WaveMultiPrefixCountBits",
    "WaveMultiPrefixProduct",
    "WaveMultiPrefixSum",
    "QuadReadLaneAt",
    "QuadReadAcrossX",
    "QuadReadAcrossY",
    "QuadReadAcrossDiagonal",
    "QuadAny",
    "QuadAll",
    "TraceRay",
    "ReportHit",
    "CallShader",
    "IgnoreHit",
    "AcceptHitAndEndSearch",
    "DispatchRaysIndex",
    "DispatchRaysDimensions",
    "WorldRayOrigin",
    "WorldRayDirection",
    "ObjectRayOrigin",
    "ObjectRayDirection",
    "RayTMin",
    "RayTCurrent",
    "PrimitiveIndex",
    "InstanceID",
    "InstanceIndex",
    "GeometryIndex",
    "HitKind",
    "RayFlags",
    "ObjectToWorld",
    "WorldToObject",
    "ObjectToWorld3x4",
    "WorldToObject3x4",
    "ObjectToWorld4x3",
    "WorldToObject4x3",
    "dot4add_u8packed",
    "dot4add_i8packed",
    "dot2add",
    "unpack_s8s16",
    "unpack_u8u16",
    "unpack_s8s32",
    "unpack_u8u32",
    "pack_s8",
    "pack_u8",
    "pack_clamp_s8",
    "pack_clamp_u8",
    "SetMeshOutputCounts",
    "DispatchMesh",
    "IsHelperLane",
    "AllocateRayQuery",
    "CreateResourceFromHeap",
    "and",
    "or",
    "select",
    // DXC resource and other types, from https://github.com/microsoft/DirectXShaderCompiler/blob/18c9e114f9c314f93e68fbc72ce207d4ed2e65ae/tools/clang/lib/AST/HlslTypes.cpp#L441-#L572
    "InputPatch",
    "OutputPatch",
    "PointStream",
    "LineStream",
    "TriangleStream",
    "Texture1D",
    "RWTexture1D",
    "Texture2D",
    "RWTexture2D",
    "Texture2DMS",
    "RWTexture2DMS",
    "Texture3D",
    "RWTexture3D",
    "TextureCube",
    "RWTextureCube",
    "Texture1DArray",
    "RWTexture1DArray",
    "Texture2DArray",
    "RWTexture2DArray",
    "Texture2DMSArray",
    "RWTexture2DMSArray",
    "TextureCubeArray",
    "RWTextureCubeArray",
    "FeedbackTexture2D",
    "FeedbackTexture2DArray",
    "RasterizerOrderedTexture1D",
    "RasterizerOrderedTexture2D",
    "RasterizerOrderedTexture3D",
    "RasterizerOrderedTexture1DArray",
    "RasterizerOrderedTexture2DArray",
    "RasterizerOrderedBuffer",
    "RasterizerOrderedByteAddressBuffer",
    "RasterizerOrderedStructuredBuffer",
    "ByteAddressBuffer",
    "RWByteAddressBuffer",
    "StructuredBuffer",
    "RWStructuredBuffer",
    "AppendStructuredBuffer",
    "ConsumeStructuredBuffer",
    "Buffer",
    "RWBuffer",
    "SamplerState",
    "SamplerComparisonState",
    "ConstantBuffer",
    "TextureBuffer",
    "RaytracingAccelerationStructure",
    // DXC templated types, from https://github.com/microsoft/DirectXShaderCompiler/blob/18c9e114f9c314f93e68fbc72ce207d4ed2e65ae/tools/clang/lib/AST/ASTContextHLSL.cpp
    // look for `BuiltinTypeDeclBuilder`
    "matrix",
    "vector",
    "TextureBuffer",
    "ConstantBuffer",
    "RayQuery",
    // Naga utilities
    super::writer::MODF_FUNCTION,
    super::writer::FREXP_FUNCTION,
    super::writer::EXTRACT_BITS_FUNCTION,
    super::writer::INSERT_BITS_FUNCTION,
    super::writer::SAMPLER_HEAP_VAR,
    super::writer::COMPARISON_SAMPLER_HEAP_VAR,
];

// DXC scalar types, from https://github.com/microsoft/DirectXShaderCompiler/blob/18c9e114f9c314f93e68fbc72ce207d4ed2e65ae/tools/clang/lib/AST/ASTContextHLSL.cpp#L48-L254
// + vector and matrix shorthands
pub const TYPES: &[&str] = &{
    const L: usize = 23 * (1 + 4 + 4 * 4);
    let mut res = [""; L];
    let mut c = 0;

    /// For each scalar type, it will additionally generate vector and matrix shorthands
    macro_rules! generate {
        ([$($roots:literal),*], $x:tt) => {
            $(
                generate!(@inner push $roots);
                generate!(@inner $roots, $x);
            )*
        };

        (@inner $root:literal, [$($x:literal),*]) => {
            generate!(@inner vector $root, $($x)*);
            generate!(@inner matrix $root, $($x)*);
        };

        (@inner vector $root:literal, $($x:literal)*) => {
            $(
                generate!(@inner push concat!($root, $x));
            )*
        };

        (@inner matrix $root:literal, $($x:literal)*) => {
            // Duplicate the list
            generate!(@inner matrix $root, $($x)*; $($x)*);
        };

        // The head/tail recursion: pick the first element of the first list and recursively do it for the tail.
        (@inner matrix $root:literal, $head:literal $($tail:literal)*; $($x:literal)*) => {
            $(
                generate!(@inner push concat!($root, $head, "x", $x));
            )*
            generate!(@inner matrix $root, $($tail)*; $($x)*);

        };

        // The end of iteration: we exhausted the list
        (@inner matrix $root:literal, ; $($x:literal)*) => {};

        (@inner push $v:expr) => {
            res[c] = $v;
            c += 1;
        };
    }

    generate!(
        [
            "bool",
            "int",
            "uint",
            "dword",
            "half",
            "float",
            "double",
            "min10float",
            "min16float",
            "min12int",
            "min16int",
            "min16uint",
            "int16_t",
            "int32_t",
            "int64_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "float16_t",
            "float32_t",
            "float64_t",
            "int8_t4_packed",
            "uint8_t4_packed"
        ],
        ["1", "2", "3", "4"]
    );

    debug_assert!(c == L);

    res
};
