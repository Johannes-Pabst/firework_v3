pub fn spectrum_file_to_rgb(path: &str) -> [f32; 3]{
    let data = std::fs::read_to_string(path).unwrap();
    let mut sum = [0.0f32; 3];
    
}