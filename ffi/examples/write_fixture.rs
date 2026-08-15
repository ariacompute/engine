//! Example: write tiny fixture to argv[1].
use aria_inference::fixture::write_tiny_q4_bundle;
use std::env;
use std::path::PathBuf;

fn main() {
    let out = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("bindings/testdata/tiny-q4"));
    std::fs::create_dir_all(&out).unwrap();
    write_tiny_q4_bundle(&out).unwrap();
    println!("{}", out.display());
}
