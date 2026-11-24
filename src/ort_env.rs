//! The process-wide ONNX Runtime environment.
//!
//! ort 2.0.0-rc.15 creates every session in an explicit [`Environment`];
//! all of voxtype's ONNX models share this one.

use ort::environment::Environment;
use ort::session::builder::SessionBuilder;
use std::sync::OnceLock;

/// The shared environment, created on first use.
pub fn env() -> ort::Result<&'static Environment> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        ort::init()
            .with_name("voxtype")
            .build()
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| ort::Error::new(e.clone()))
}

/// A session builder in the shared environment.
pub fn session_builder() -> ort::Result<SessionBuilder> {
    ort::session::Session::builder(env()?)
}
