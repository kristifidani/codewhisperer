use tracing_subscriber::EnvFilter;

/// Initializes the global tracing subscriber for logging.
///
/// This function sets up a `tracing_subscriber` with a format layer and an environment
/// filter. The environment filter reads the `RUST_LOG` environment variable to determine
/// the log level and filtering rules.
pub fn tracing_setup() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()) // reads RUST_LOG
        .with_target(true)
        .try_init()
        .unwrap_or_else(|_| tracing::warn!("Tracing subscriber already initialized"));
}
