//! File locking for distributed processing.

mod file_lock;

pub use file_lock::{cleanup_all_locks, FileLock, LockInfo};
