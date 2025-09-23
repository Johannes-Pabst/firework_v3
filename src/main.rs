use std::process::Stdio;

use tokio::process::Command;

pub mod renderer;
pub mod colors;
pub mod instructions_helper;

pub const WIDTH:u32=3840;
pub const HEIGHT:u32=2160;

pub const TW:u32=2000;

#[tokio::main]
async fn main() {
    let mut state = renderer::prepare().await;
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
    for i in 0..120{
        renderer::render(&mut state, stdin).await;
        println!("{i}");
    }
    ffmpeg.wait().await.unwrap();
}
