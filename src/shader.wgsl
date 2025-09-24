struct Screen {
    size: vec2<f32>, // width, height
    frame: u32,
    padding: f32,
};

struct GpuCurve{// no buffer!
    start:u32,
    end:u32,
    align:u32,
}
struct CurvePoint{// const buffer!
    t:u32,
    v:f32,
    buffer:vec2<f32>,
}
struct ParticleInstructions{// const buffer!
    friction:GpuCurve,
    v_thruster_spawner:u32,
    v_thruster_strength:GpuCurve,
    c_thruster_spawner:u32,
    c_thruster_strength:GpuCurve,
    passive_spawner:u32,
    r:GpuCurve,
    buffer1:u32,
    g:GpuCurve,
    buffer2:u32,
    b:GpuCurve,
    buffer3:u32,
}
struct Spawner{// const buffer!
    strength:GpuCurve,
    max_v:f32,
    alive_fraction:GpuCurve,
    min_v:f32,
    instruction:u32,
    next_spawner:u32,
    skip_prob:f32,
    variance:f32,
}

@group(0) @binding(0)
var<uniform> screen: Screen;

@group(1) @binding(0) var bw_tex: texture_2d<f32>;
@group(1) @binding(1) var bw_sampler: sampler;

@group(2) @binding(0) var<storage, read> input_particles: array<FlareDataPlain>;
@group(2) @binding(1) var<storage, read_write> output_particles: array<FlareDataPlain>;
@group(2) @binding(2) var<storage, read_write> counter: atomic<u32>;
@group(2) @binding(3) var<storage, read> instructions: array<ParticleInstructions>;
@group(2) @binding(4) var<storage, read> spawners: array<Spawner>;
@group(2) @binding(5) var<storage, read> curves: array<CurvePoint>;

struct FlareDataPlain {
    pos: vec3<f32>,
    instruction:u32,
    v: vec3<f32>,
    lifetime:u32,
    color: vec3<f32>,
    start_time:u32,
    cthruster_dir:vec3<f32>,
    buffer1:f32,
    cthruster_perp:vec3<f32>,
    buffer2:f32,
}
struct FlareData {
    @location(0) pos: vec3<f32>,
    @location(1) instruction:u32,
    @location(2) v: vec3<f32>,
    @location(3) lifetime:u32,
    @location(4) color: vec3<f32>,
    @location(5) start_time:u32,
    @location(6) cthruster_dir:vec3<f32>,
    @location(7) buffer1:f32,
    @location(8) cthruster_perp:vec3<f32>,
    @location(9) buffer2:f32,
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

fn hash44(p4:vec4<f32>)->vec4<f32>
{
	var p5 = fract(p4  * vec4<f32>(.1031, .1030, .0973, .1099));
    p5 += dot(p5, p5.wzxy+33.33);
    return fract((p5.xxyz+p5.yzzw)*p5.zywx);
}// code from https://www.shadertoy.com/view/4djSRW

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    var pospx=rel_to_screen(vertex.pos.xy);
    var gf=glare_falloff(distance(pospx,vertex.position.xy), vertex.radius/2)*(textureSampleMiddleDetail(bw_tex, bw_sampler, (vertex.position.xy-pospx)/500+vec2<f32>(0.5)).r+0.3);
    // if(gf==0.0){
    //     return vec4<f32>(1.0,0.0,0.0,1.0);
    // }
    return vec4<f32>(gf*vertex.color.xyz-(hash44(vec4<f32>(vertex.position.xy, vertex.pos.xy))).xyz/255.0, 1.0);
}

fn sample_curve(c:GpuCurve, t:u32)->f32{
    if(c.start>c.end){
        return 0.0;
    }
    if(c.start==c.end){
        return curves[c.start].v;
    }
    if(t>=curves[c.end].t){
        return curves[c.end].v;
    }
    var prev=c.start;
    while(curves[prev].t<=t && prev<c.end-1){
        prev=prev+1u;
    }
    if(prev==c.start){
        return curves[c.start].v;
    }
    return curves[prev-1u].v*(f32(curves[prev].t-t)/f32(curves[prev].t-curves[prev-1u].t))
        +curves[prev].v*(f32(t-curves[prev-1u].t)/f32(curves[prev].t-curves[prev-1u].t));
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&input_particles)) {
        return;
    }

    var p = input_particles[idx];

    p.pos += p.v * 0.0016;
    p.v += vec3<f32>(0.0, -9.8, 0.0) * 0.016;
    var st = screen.frame-p.start_time-p.lifetime;
    p.v*=sample_curve(instructions[p.instruction].friction, st);
    p.color=vec3<f32>(sample_curve(instructions[p.instruction].r, st),sample_curve(instructions[p.instruction].g, st),sample_curve(instructions[p.instruction].b, st));

    // c_thruster_spawner
    if(instructions[p.instruction].c_thruster_spawner!=0xffffffffu){
        var sp=spawners[instructions[p.instruction].c_thruster_spawner];
        var strength=sample_curve(sp.strength, st);
        var target_bevore=u32(strength*f32(st));
        var target_after=u32(strength*f32(st+1u));
        var to_spawn=target_after-target_bevore;
        for(var i=0u; i<to_spawn; i=i+1u){
            let iid = atomicAdd(&counter, 1u);
            output_particles[iid] = FlareDataPlain(
                p.pos,
                sp.instruction,
                p.v - p.cthruster_dir,
                200u,
                vec3<f32>(0.0, 0.0, 0.0),
                screen.frame,
                p.cthruster_dir,
                0.0,
                p.cthruster_perp,
                0.0
            );
        }
    }

    if (p.lifetime+p.start_time-screen.frame == 0) {
        return;
    }
    let id = atomicAdd(&counter, 1u);
    output_particles[id] = p;
}