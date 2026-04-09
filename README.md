# Fireworks renderer
an elaborate algorithm to render AND simulate millions of unnecessarily detailed flares which try to mimic artifacts which only appear in the eye when looking at bright spots on the GPU using webGPU vertex, fragment and compute shaders. The result is encoded into an mp4 using ffmpeg. download output.mp4 to view the current state of the simulation.
## screenshot
![screenshot](image.png)

## rendering flares
all flares are rendered using the same 2000 by 2000 grayscale texture. It is generated at the start of the program using 3d perlin noise octaves indexed in a cone. This helps produce the rays seen in the screenshot. The texture is then sampled in a way to concentrate the detail to the center of the flare and use a lower resolution at the dark edges. 

## simulating flares
Flare motion is simulated using a compute shader. This maximizes simulation speed and reduces the data sent between CPU and GPU to near zero. Each flare saves an index to a ParticleInstructions object which manages information like air friction, thrusters, color and even how and when it spawns other flares. All (or at least most) parameters are controlled using GpuCurve Objects which allow the parameters to follow a set of linear increase/decrease sections over time to e.g. fade in brightness or make a rocket's thruster stop bevore it explodes. ParticleInstructions objects are generated on the CPU and allow for quite powerful customization of a flare's behavior without having to alter any wgsl code.
