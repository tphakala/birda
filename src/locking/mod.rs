//! File locking for distributed processing and config serialisation.

mod config_lock;
mod file_lock;

pub use config_lock::{cleanup_all_config_locks, with_config_lock};
pub use file_lock::{FileLock, LockInfo, cleanup_all_locks};
