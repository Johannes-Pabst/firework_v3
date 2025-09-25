use std::{collections::HashMap, ops::Mul};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCurve{// no buffer!
    pub _start:u32,
    pub _end:u32,
    pub _align:u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CurvePoint{// const buffer!
    pub _t:f32,
    pub _v:f32,
    pub _buffer:[f32;2],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleInstructions{// const buffer!
    pub _friction:GpuCurve,
    pub _v_thruster_spawner:u32,
    pub _v_thruster_strength:GpuCurve,
    pub _c_thruster_spawner:u32,
    pub _c_thruster_strength:GpuCurve,
    pub _passive_spawner:u32,
    pub _r:GpuCurve,
    pub _buffer1:u32,
    pub _g:GpuCurve,
    pub _buffer2:u32,
    pub _b:GpuCurve,
    pub _buffer3:u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Spawner{// const buffer!
    pub _strength:GpuCurve,
    pub _max_v:f32,
    pub _alive_fraction:GpuCurve,
    pub _min_v:f32,
    pub _instruction:u32,
    pub _next_spawner:u32,
    pub _skip_prob:f32,
    pub _variance:f32,
}
#[derive(Clone)]
pub struct Curve{
    pub points:Vec<CurvePoint>,
    pub align:u32,
}
impl CurvePoint{
    pub fn new(t:f32,v:f32)->Self{
        CurvePoint{_t:t,_v:v,_buffer:[0.0,0.0]}
    }
}
impl Mul<f32> for Curve{
    type Output=Curve;
    fn mul(mut self, rhs:f32)->Self::Output{
        for p in self.points.iter_mut(){
            p._v*=rhs;
        }
        self
    }
}
impl Mul<[f32;3]> for Curve{
    type Output=ColCurve;
    fn mul(self, rhs:[f32;3])->Self::Output{
        ColCurve{
            r:self.clone()*rhs[0],
            g:self.clone()*rhs[1],
            b:self*rhs[2],
        }
    }
}
impl Curve{
    pub fn new(mut points:Vec<CurvePoint>, align:u32)->Self{
        points.sort_by(|a,b|a._t.partial_cmp(&b._t).unwrap());
        Curve { points, align }
    }
    pub fn fr_cst(v:f32)->Self{
        Curve::new(vec![CurvePoint{_t:0.0,_v:v,_buffer:[0.0,0.0]}], 0)
    }
    pub fn zero()->Self{
        Self::fr_cst(0.0)
    }
}
pub struct ColCurve{
    pub r:Curve,
    pub g:Curve,
    pub b:Curve,
}
pub struct HelperElement<T>{
    pub i:u32,
    pub e:T,
}
pub struct Helper<T>{
    pub ids:HashMap<String,u32>,
    pub data:Vec<T>,
}
pub trait ToCurve{
    fn to_curve(&self,buf:&mut Vec<CurvePoint>)->GpuCurve;
}
impl ToCurve for GpuCurve{
    fn to_curve(&self,_buf:&mut Vec<CurvePoint>)->GpuCurve{
        self.clone()
    }
}
impl ToCurve for f32{
    fn to_curve(&self,buf:&mut Vec<CurvePoint>)->GpuCurve{
        GpuCurve::fr_cst(buf, *self)
    }
}
impl ToCurve for Curve{
    fn to_curve(&self,buf:&mut Vec<CurvePoint>)->GpuCurve{
        GpuCurve::fr_cur(buf, self.clone())
    }
}
impl<T> Helper<T>{
    pub fn save(&mut self,name:&str,e:T)->u32{
        let i = self.data.len() as u32;
        self.ids.insert(name.to_string(), i);
        self.data.push(e);
        i
    }
    pub fn load(&self, name:&str)->u32{
        *self.ids.get(&name.to_string()).unwrap()
    }
    pub fn new()->Self{
        Helper { ids: HashMap::new(), data: Vec::new() }
    }
}
impl GpuCurve{
    pub fn new(buf:&mut Vec<CurvePoint>, mut points:Vec<CurvePoint>, align:u32)->Self{
        let start=buf.len() as u32;
        buf.extend(points.drain(..));
        let end=(buf.len()-1) as u32;
        println!("start: {start}, end: {end}");
        GpuCurve { _start: start, _end: end, _align: align }
    }
    pub fn fr_cur(buf:&mut Vec<CurvePoint>, curve:Curve)->Self{
        GpuCurve::new(buf, curve.points, curve.align)
    }
    pub fn fr_cst(buf:&mut Vec<CurvePoint>, v:f32)->Self{
        Self::new(buf, vec![CurvePoint{_t:0.0, _v:v,_buffer:[0.0,0.0]}], 0)
    }
    pub fn zero(buf:&mut Vec<CurvePoint>)->Self{
        Self::fr_cst(buf, 0.0)
    }
}
impl ParticleInstructions{
    pub fn new<C>(c_buf:&mut Vec<CurvePoint>, friction:C, color:ColCurve)->Self where C:ToCurve{
        ParticleInstructions { _friction: friction.to_curve(c_buf), _v_thruster_spawner: u32::MAX, _v_thruster_strength: GpuCurve::zero(c_buf), _c_thruster_spawner: u32::MAX, _c_thruster_strength: GpuCurve::zero(c_buf), _passive_spawner: u32::MAX, _r: GpuCurve::fr_cur(c_buf, color.r), _buffer1: 0, _g: GpuCurve::fr_cur(c_buf, color.g), _buffer2: 0, _b: GpuCurve::fr_cur(c_buf, color.b), _buffer3: 0 }
    }
    pub fn with_v_thruster<C>(mut self,c_buf:&mut Vec<CurvePoint>, strength:C)->Self where C:ToCurve{
        self._v_thruster_strength=strength.to_curve(c_buf);
        self
    }
    pub fn with_c_thruster<C>(mut self,c_buf:&mut Vec<CurvePoint>, strength:C)->Self where C:ToCurve{
        self._c_thruster_strength=strength.to_curve(c_buf);
        self
    }
    pub fn with_passive_spawner(mut self,spawner:u32)->Self{
        self._passive_spawner=spawner;
        self
    }
    pub fn with_v_thruster_spawner(mut self,spawner:u32)->Self{
        self._v_thruster_spawner=spawner;
        self
    }
    pub fn with_c_thruster_spawner(mut self,spawner:u32)->Self{
        self._c_thruster_spawner=spawner;
        self
    }
}
impl Spawner{
    pub fn new<C1,C2>(c_buf:&mut Vec<CurvePoint>, strength:C1, alive_fraction:C2, variance:f32, min_v:f32, max_v:f32, instruction:u32)->Self where C1:ToCurve, C2:ToCurve{
        Spawner { _strength: strength.to_curve(c_buf), _max_v: max_v, _alive_fraction: alive_fraction.to_curve(c_buf), _min_v: min_v, _instruction: instruction, _next_spawner: u32::MAX, _skip_prob: 0.0, _variance: variance }
    }
    pub fn with_next_spawner(mut self, spawner:u32, skip_prob:f32)->Self{
        self._next_spawner=spawner;
        self._skip_prob=skip_prob;
        self
    }
}