use std::collections::HashMap;

pub struct Curve{// no buffer!
    pub _start:u32,
    pub _end:u32,
    pub _align:u32,
}
pub struct CurvePoint{// const buffer!
    pub _t:u32,
    pub _v:f32,
}
pub struct ParticleInstructions{// const buffer!
    pub _friction:Curve,
    pub _v_thruster_spawner:u32,
    pub _v_thruster_strength:Curve,
    pub _c_thruster_spawner:u32,
    pub _c_thruster_strength:Curve,
    pub _passive_spawner:u32,
    pub _r:Curve,
    pub _buffer1:u32,
    pub _g:Curve,
    pub _buffer2:u32,
    pub _b:Curve,
    pub _buffer3:u32,
}
pub struct Spawner{// const buffer!
    pub _strength:Curve,
    pub _max_v:f32,
    pub _alive_fraction:Curve,
    pub _min_v:f32,
    pub _instruction:u32,
    pub _next_spawner:u32,
    pub _skip_prob:f32,
    pub _buffer1:u32,
}
pub struct HelperElement<T>{
    pub i:u32,
    pub e:T,
}
pub struct Helper<T>{
    pub ids:HashMap<String,u32>,
    pub data:Vec<T>,
}
impl<T> Helper<T>{
    pub fn save(&mut self,name:&str,e:T){
        self.ids.insert(name.to_string(), self.data.len() as u32);
        self.data.push(e);
    }
    pub fn load(&self, name:&str)->u32{
        *self.ids.get(&name.to_string()).unwrap()
    }
    pub fn new()->Self{
        Helper { ids: HashMap::new(), data: Vec::new() }
    }
}
impl Curve{
    pub fn new(buf:&mut Vec<CurvePoint>, mut points:Vec<CurvePoint>, align:u32)->Self{
        let start=buf.len() as u32;
        buf.extend(points.drain(..));
        let end=(buf.len()-1) as u32;
        Curve { _start: start, _end: end, _align: align }
    }
    pub fn fr_cst(buf:&mut Vec<CurvePoint>, v:f32)->Self{
        Self::new(buf, vec![CurvePoint{_t:0, _v:v}], 0)
    }
}
impl ParticleInstructions{
    pub fn new(friction:Curve, )
}