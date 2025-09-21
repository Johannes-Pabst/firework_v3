struct Screen {
    size: vec2<u32>, // width, height
};

@group(0) @binding(0)
var<uniform> screen: Screen;

@group(1) @binding(0) var bw_tex: texture_2d<f32>;
@group(1) @binding(1) var bw_sampler: sampler;

struct FlareData {
    @location(0) pos: vec3<f32>,
    @location(1) instruction:u32,
    @location(2) v: vec3<f32>,
    @location(3) lifetime:u32,
    @location(4) color: vec3<f32>,
    @location(5) buffer1:f32,
    @location(6) cthruster_dir:vec3<f32>,
    @location(7) buffer2:f32,
    @location(8) cthruster_perp:vec3<f32>,
    @location(9) buffer3:f32,
}
struct Vertex {
    @location(10) pos: vec4<f32>,
}
struct VertexOutput {
    @location(0) pos: vec3<f32>,
    @location(1) buffer:f32,
    @location(2) color: vec3<f32>,
    @location(3) radius:f32,// in px
    @builtin(position) position: vec4<f32>,
}

const fexp=1.6;

fn glare_falloff_raw(dist:f32)->f32{
    return pow(dist,-fexp);
}
fn glare_falloff_inverse(falloff:f32)->f32{
    return pow(falloff,1/-fexp);
}
fn glare_falloff(dist:f32, radius:f32)->f32{
    return max(0.0,glare_falloff_raw(dist)-glare_falloff_raw(radius));
}

fn rel_to_screen(in:vec2<f32>)->vec2<f32>{
    return (in*vec2<f32>(1, -1)+vec2<f32>(1, 1))/2*vec2<f32>(screen.size);
}

fn textureSampleMiddleDetail(tex: texture_2d<f32>, s: sampler, pos: vec2<f32>)->vec4<f32>{
    var len=length(pos-vec2<f32>(0.5));
    return textureSample(tex, s, (pos-vec2<f32>(0.5))/(1.0+len*2.0*2.0)+vec2<f32>(0.5));
}

@vertex
fn vs_main(
    fd:FlareData, v:Vertex
) -> VertexOutput {
    var min_brightness=0.001;
    var result: VertexOutput;
    result.color = fd.color/fd.pos.z/fd.pos.z;
    var brightness=result.color.x+result.color.y+result.color.z;
    result.radius = glare_falloff_inverse(min_brightness/brightness);
    result.pos = fd.pos/fd.pos.z;
    result.position = vec4<f32>(v.pos.xy/vec2<f32>(screen.size)*vec2<f32>(result.radius,result.radius)+fd.pos.xy/fd.pos.z, 0.0, 1.0);
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    var pospx=rel_to_screen(vertex.pos.xy);
    var gf=glare_falloff(distance(pospx,vertex.position.xy), vertex.radius/2)*(textureSampleMiddleDetail(bw_tex, bw_sampler, (vertex.position.xy-pospx)/500+vec2<f32>(0.5)).r+0.3);
    // if(gf==0.0){
    //     return vec4<f32>(1.0,0.0,0.0,1.0);
    // }
    return vec4<f32>(gf*vertex.color.xyz, 1.0);
}