# NSIS Evaluation Result

**Date:** 2026-02-10
**Conclusion:** NSIS is not suitable for Birda installer
**Recommendation:** Continue using Inno Setup

## Problem

NSIS is a 32-bit application with a 2GB memory limit. Birda's CUDA distribution includes 2.4GB of DLLs (some individual files are 667MB), which causes NSIS to fail with internal compiler error #12345 during the build process:

```
Internal compiler error #12345: error mmapping file (2093984594, 33554432) is out of range.
```

## Attempted Solutions

Multiple approaches were tried to work around NSIS limitations:

1. **Conditional PRODUCT_VERSION definition** - Fixed redefinition errors
2. **Working directory approach** - Changed from `/NOCD` to proper working directory
3. **Explicit DLL listing** - Replaced wildcards with explicit file lists
4. **Disabled compression** - Attempted to avoid compression memory overhead (ignored by `/SOLID` mode)

All attempts failed with the same memory mapping error when processing large CUDA DLLs.

## Root Cause

NSIS is fundamentally limited by its 32-bit architecture:
- Maximum addressable memory: ~2GB
- CUDA DLLs total size: 2.4GB
- Largest individual DLL: 667MB (cublasLt64_12.dll)

Even with compression disabled, NSIS must memory-map files during processing, and the large CUDA libraries exceed its capacity.

## Comparison with Inno Setup

**Inno Setup:**
- Successfully handles 2.4GB of CUDA DLLs
- Reliable compression and packaging
- Currently working in production releases
- No memory limit issues

**NSIS:**
- Cannot process files due to 32-bit limitations
- Multiple failed workaround attempts
- Would require architectural changes (split installers, download-on-demand) to work

## Recommendation

**Continue using Inno Setup** for Birda's Windows installer. It handles the large CUDA distribution without issues and has proven reliable in production.

NSIS would only be viable if:
1. CUDA libraries were downloaded separately during installation (not bundled)
2. A 64-bit NSIS compiler became available (currently doesn't exist)
3. The CUDA library size was significantly reduced

None of these options are practical for Birda's current distribution model.
