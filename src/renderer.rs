use std::{
    io::{Stdout, Write, stdout},
    process::Stdio,
};

use noise::NoiseFn;
use rand::Rng;
use tokio::{io::AsyncWriteExt, process::Command};
use wgpu::util::DeviceExt;

use crate::{HEIGHT, TW, WIDTH, colors::wavelength_to_stimul};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Screen {
    _size: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    _pos: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlareData {
    _pos: [f32;3],
    _instruction:u32,
    _v: [f32;3],
    _lifetime:u32,
    _color: [f32;3],
    _buffer1:f32,
    _cthruster_dir:[f32;3],
    _buffer2:f32,
    _cthruster_perp:[f32;3],
    _buffer3:f32,
}

fn vertex(pos: [f32; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], 0.0, 1.0],
    }
}

fn quad(vertices: &mut Vec<Vertex>, indices: &mut Vec<u16>) {
    let base = vertices.len() as u16;

    vertices.extend_from_slice(&[
        vertex([-1.0, -1.0]),
        vertex([1.0, -1.0]),
        vertex([1.0, 1.0]),
        vertex([-1.0, 1.0]),
    ]);

    indices.extend([0, 1, 2, 2, 3, 0].iter().map(|i| base + *i));
}

fn create_vertices() -> (Vec<Vertex>, Vec<FlareData>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut flares = Vec::new();
    let mut indices = Vec::new();

    quad(&mut vertices, &mut indices);
    // quad(&mut vertices, &mut indices, blue, -0.5);

    flares.push(FlareData {
        _pos: [-0.01, -0.01, 0.1],
        _color: wavelength_to_stimul(500.0),
        _instruction: u32::MAX,
        _v: [0.0, 0.0, 0.0],
        _lifetime: 60,
        _cthruster_dir: [0.0, 0.0, 0.0],
        _cthruster_perp: [0.0, 0.0, 0.0],
        _buffer1: 0.0,
        _buffer2: 0.0,
        _buffer3: 0.0,
    });


    (vertices, flares, indices)
}

pub struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,

    vertex_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,

    instance_count: u32,
}

pub async fn prepare() -> State {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            ..Default::default()
        })
        .await
        .unwrap();

    let texture_data = Vec::<u8>::with_capacity((WIDTH * HEIGHT * 4) as usize);
    let render_target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: texture_data.capacity() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let texture_view: wgpu::TextureView =
        render_target.create_view(&wgpu::TextureViewDescriptor::default());

    // Create the vertex and index buffers
    let vertex_size = size_of::<Vertex>();
    let flare_data_size = size_of::<FlareData>();
    let (vertex_data, instance_data, index_data) = create_vertices();

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(&instance_data),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsages::INDEX,
    });

    let screen_uniform = Screen {
        _size: [WIDTH, HEIGHT],
    };

    let screen_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Screen Uniform Buffer"),
        contents: bytemuck::bytes_of(&screen_uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create pipeline layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Globals Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::all(),
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let width = TW;
    let height = TW;

    let fx = width / 2;
    let fy = height / 2;

    let mut data: Vec<u16> = vec![0u16; (width * height) as usize];

    let mut rng = rand::rng();

    let noise = noise::OpenSimplex::new(rng.random());

    let mut max: f64 = 0.0;
    let mut min: f64 = 1.0;

    for x in 0..width {
        for y in 0..height {
            let d1 = ((fx as f64 - x as f64).powi(2) + (fy as f64 - y as f64).powi(2)).sqrt();
            let rx = (x as f64 - fx as f64) / (1.0 + d1 / fx as f64 * 2.0) + fx as f64;
            let ry = (y as f64 - fy as f64) / (1.0 + d1 / fy as f64 * 2.0) + fy as f64;
            let distance =
                ((fx as f64 - rx as f64).powi(2) + (fy as f64 - ry as f64).powi(2)).sqrt();
            let mut n1;
            if distance < 0.1 {
                n1 = 1.0;
            } else {
                let dx =
                    ((rx as f64 + rng.random_range(-1.0..1.0) * 0.1) - fx as f64) / distance as f64;
                let dy =
                    ((ry as f64 + rng.random_range(-1.0..1.0) * 0.1) - fy as f64) / distance as f64;

                n1 = (noise.get([dx * 25.0, dy * 25.0, distance * 3.0 * 0.01])
                    + noise.get([dx * 50.0, dy * 50.0, distance * 3.0 * 0.02])
                    + noise.get([dx * 100.0, dy * 100.0, distance * 3.0 * 0.04])
                    + noise.get([dx * 200.0, dy * 200.0, distance * 3.0 * 0.08])
                    + noise.get([dx * 400.0, dy * 400.0, distance * 3.0 * 0.16])
                    + noise.get([dx * 800.0, dy * 800.0, distance * 3.0 * 0.32])
                    + noise.get([dx * 1600.0, dy * 1600.0, distance * 3.0 * 0.64])
                    + noise.get([dx * 3200.0, dy * 3200.0, distance * 3.0 * 1.28]))
                    / 8.0
                    / 2.0
                    * 2.9
                    + 0.5;
                max = max.max(n1);
                min = min.min(n1);
            }
            n1 = n1.max(0.0).min(1.0);
            data[(x + y * width) as usize] = ((n1 * (256.0 * 256.0 - 0.01)).floor()) as u16;
        }
    }

    println!("max: {max}, min: {min}");

    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BW Texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Unorm, // single channel [0,1]
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 2),
            rows_per_image: Some(height),
        },
        texture_size,
    );

    let bw_texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("BW Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

    let texture_bind_group: wgpu::BindGroup =
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bw_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout, &texture_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Globals Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: screen_buf.as_entire_binding(),
        }],
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let vertex_buffers = [
        wgpu::VertexBufferLayout {
            array_stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![4=>Float32x4],
        },
        wgpu::VertexBufferLayout {
            array_stride: flare_data_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0=>Float32x3, 1=>Float32, 2=>Float32x3, 3=>Float32],
        },
    ];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &vertex_buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                        dst_factor: wgpu::BlendFactor::One,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                        dst_factor: wgpu::BlendFactor::One,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    let state = State {
        device,
        queue,
        texture: render_target,

        vertex_buf,
        instance_buf,
        index_buf,
        output_buf,
        index_count: index_data.len(),
        bind_group,
        texture_bind_group,
        pipeline,

        instance_count: instance_data.len() as u32,
    };
    state
}
pub async fn render(state: &mut State) {
    let texture_view = state.texture.create_view(&wgpu::TextureViewDescriptor {
        // Without add_srgb_suffix() the image we will be working with
        // might not be "gamma correct".
        format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        ..Default::default()
    });

    state.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.push_debug_group("Prepare data for draw.");
        rpass.set_pipeline(&state.pipeline);
        rpass.set_bind_group(0, &state.bind_group, &[]);
        rpass.set_bind_group(1, &state.texture_bind_group, &[]);
        rpass.set_index_buffer(state.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        rpass.set_vertex_buffer(0, state.vertex_buf.slice(..));
        rpass.set_vertex_buffer(1, state.instance_buf.slice(..));
        rpass.pop_debug_group();
        rpass.insert_debug_marker("Draw!");
        rpass.draw_indexed(0..state.index_count as u32, 0, 0..state.instance_count);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &state.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &state.output_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                // This needs to be a multiple of 256. Normally we would need to pad
                // it but we here know it will work out anyways.
                bytes_per_row: Some((WIDTH * 4) as u32),
                rows_per_image: Some(HEIGHT as u32),
            },
        },
        wgpu::Extent3d {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            depth_or_array_layers: 1,
        },
    );

    state.queue.submit(Some(encoder.finish()));

    let buffer_slice = state.output_buf.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
    state.device.poll(wgpu::wgt::PollType::Wait).unwrap();
    let data = buffer_slice.get_mapped_range();
    let mut ffmpeg = Command::new("ffmpeg")
        .args(&[
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{}x{}", WIDTH, HEIGHT),
            "-r",
            "60", // FPS
            "-i",
            "-", // read from stdin
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "output.mp4",
        ])
        .stdin(Stdio::piped())
        .spawn()
        .expect("Failed to spawn ffmpeg");

    let stdin = ffmpeg.stdin.as_mut().unwrap();
    println!("Video saved to output.mp4");
    println!("Video saved to output.mp4");
    println!("Video saved to output.mp4");
    println!("Video saved to output.mp4");
    stdin.write_all(&data).await.unwrap();
    ffmpeg.wait().await.unwrap();
}
