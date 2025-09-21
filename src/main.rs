pub mod renderer;
pub mod colors;

pub const WIDTH:u32=3840;
pub const HEIGHT:u32=2160;

pub const TW:u32=2000;

#[tokio::main]
async fn main() {
    let mut state = renderer::prepare().await;
    renderer::render(&mut state).await;
}
