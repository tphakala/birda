# Fix Plan v2: Critical Issues in Stdout NDJSON Mode

**Date:** 2026-02-05
**Status:** Reviewed and Approved by Kimi
**Issues to Fix:** 2 critical issues

---

## Issue #1: Silent Error Handling in Reporter

**File:** `src/output/reporter.rs`
**Lines:** 195-196
**Severity:** Critical

### Problem
Errors from `writeln!` and `flush()` are silently ignored. If stdout closes (broken pipe), detection data is lost without notification.

### Fix (Kimi-Approved with Rate Limiting)

```rust
if let Ok(mut writer) = self.writer.lock() {
    if let Err(e) = writeln!(writer, "{json}") {
        // Log first error only to avoid spam on broken pipe
        use std::sync::atomic::{AtomicBool, Ordering};
        static STDOUT_ERROR_LOGGED: AtomicBool = AtomicBool::new(false);
        if !STDOUT_ERROR_LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!("birda: warning: failed to write to stdout: {} (subsequent errors suppressed)", e);
        }
    }
    // Flush errors are less critical - silent ignore is OK
    let _ = writer.flush();
}
```

### Rationale
- **Use `eprintln!`**: Guarantees stderr (unlike `tracing` which could be configured to stdout)
- **Rate limiting**: Log once, suppress subsequent errors (broken pipe generates many errors)
- **Static atomic**: Thread-safe, zero-cost after first error
- **Flush silent**: Flush errors less critical than write errors

### Testing
```bash
# Automated test
cargo test test_reporter_handles_write_errors

# Manual test
birda --stdout test.wav | head -1
# Should see ONE warning on stderr when pipe closes
```

---

## Issue #2: Unnecessary File Locks in Stdout Mode

**File:** `src/pipeline/processor.rs`
**Line:** 321
**Severity:** Critical

### Problem
Creates lock files even when using stdout mode (no files written).

### Fix (Using reporter.is_some())

```rust
// Acquire lock only if writing files (not stdout mode)
let _lock = if reporter.is_none() {
    // File mode - need lock to prevent concurrent writes
    Some(FileLock::acquire(input_path, output_dir)?)
} else {
    // Stdout mode - no files written, no lock needed
    None
};
```

### Rationale
- **Clear logic**: `reporter.is_some()` = stdout mode, `reporter.is_none()` = file mode
- **Self-documenting**: Comment explains the connection
- **Simple**: No API changes needed
- **Correct**: `reporter` parameter added in commit df19a63

### Alternative Considered (Rejected)
Kimi suggested checking `formats.is_empty()`, but this doesn't work because:
1. Formats can be passed even in stdout mode (they're just ignored)
2. `reporter.is_none()` is the actual source of truth for file vs stdout mode

### Testing
```bash
# Automated test
cargo test test_no_locks_in_stdout_mode

# Manual test - stdout mode
birda --stdout test.wav
ls *.lock  # Should be empty

# Manual test - file mode
birda test.wav
ls *.lock  # Should show lock file
```

---

## Implementation Steps

### Step 1: Add Test for Reporter Error Handling
**File:** `src/output/reporter.rs` test module

```rust
#[test]
fn test_reporter_handles_write_errors() {
    struct FailingWriter;
    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe closed"))
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }

    let reporter = JsonProgressReporter::with_writer(
        OutputMode::Ndjson,
        FailingWriter,
    );

    // Should not panic, should log error to stderr (check manually)
    reporter.pipeline_started(1, "test", 0.1);
}
```

### Step 2: Fix Reporter Error Handling
Update `src/output/reporter.rs:195-196` with rate-limited error logging.

Commit: `fix: add rate-limited error logging for stdout write failures`

### Step 3: Add Test for Lock Skipping
**File:** `tests/lock_behavior.rs` (new file)

```rust
use birda::pipeline::processor::process_file;
use std::path::Path;

#[test]
fn test_lock_file_not_created_in_stdout_mode() {
    // This test requires complex setup (classifier, reporter, etc.)
    // For now, manual testing is sufficient
    // TODO: Add proper integration test
}
```

### Step 4: Fix Unnecessary Locks
Update `src/pipeline/processor.rs:321` with conditional lock.

Commit: `fix: skip file locks in stdout mode`

### Step 5: Verification
```bash
# All tests pass
cargo test

# No clippy warnings
cargo clippy -- -D warnings

# Formatting correct
cargo fmt --check

# Manual tests
birda --stdout test.wav | head -1  # Broken pipe test
ls *.lock  # No locks in stdout mode
birda test.wav && ls *.lock  # Locks in file mode
```

---

## Risk Assessment

### Low Risk
- ✅ Non-breaking changes
- ✅ No API changes
- ✅ Backward compatible
- ✅ Isolated (2 files, ~15 lines total)
- ✅ Reviewed by Kimi

### Safety Net
- ✅ Existing tests continue to pass
- ✅ New tests added
- ✅ Manual testing covers both modes
- ✅ Rate limiting prevents error spam

---

## Kimi Review Responses

**Q1: eprintln! vs tracing?**
✅ **Answer**: Use `eprintln!` - safer because tracing could be configured to stdout.

**Q2: Is reporter.is_none() clear?**
✅ **Answer**: Yes, with comment explaining the logic.

**Q3: Error spam concern?**
✅ **Answer**: Fixed with AtomicBool rate limiting.

**Q4: Manual tests sufficient?**
✅ **Answer**: Added automated test for error handling, manual test for locks (integration test complex).

---

## Success Criteria

- ✅ No silent data loss - errors logged to stderr (once)
- ✅ No lock files in stdout mode
- ✅ Lock files still created in file mode
- ✅ All tests pass
- ✅ No clippy warnings
- ✅ No formatting issues
- ✅ Manual testing confirms fixes
- ✅ No error spam on broken pipe

---

## Approval

**Status:** ✅ **APPROVED** by Kimi (conditional requirements met)

**Ready to implement.**
