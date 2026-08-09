# Lokum Engine CI/CD Fix Plan

**Goal:** Resolve the persistent GitHub Actions failure during the `Install dependencies` step for macOS-14 runners and ensure the CI pipeline accurately installs MLX and FAISS wheels.

**Current State Analysis:**
- The CI workflow fails at `Install dependencies` across all Python versions (3.10, 3.11, 3.12).
- The `SYSTEM_VERSION_COMPAT: "0"` trick was attempted to force the macOS runner to report its true version (macOS 14) so `pip` would fetch the pre-built `macosx_14_0_arm64` wheels for `mlx` and `faiss-cpu`.
- Despite fixing the YAML syntax, the pipeline still fails with `exit code 1`. Because the logs are masked inside GitHub's interface, we are flying blind regarding the exact `pip` error.

**Proposed Changes:**
1. **Enable Verbose Logging:** Update `.github/workflows/ci.yml` to run `pip install -v -e .` so that the exact reason for the failure (whether it's a missing wheel, a build failure, or a dependency conflict) is printed to the logs.
2. **Set Deployment Target:** Add `MACOSX_DEPLOYMENT_TARGET: "14.0"` explicitly. Sometimes GitHub's Python binaries are hardcoded to `11.0`, and setting this environment variable explicitly tells `pip` that `macosx_14_0` wheels are safe to install.
3. **Pre-install Core MLX:** Explicitly run `pip install mlx faiss-cpu` before `pip install -e .` to isolate if the issue is coming from `setup.py` resolving dependencies or from the packages themselves.

**Assumptions & Decisions:**
- The issue is purely related to `pip`'s wheel tag resolution on GitHub Actions runners, not a bug in `lokum-engine`'s source code.
- macOS 14+ works perfectly on real machines (like the user's M5 Mac on macOS 26); the issue is strictly isolated to the CI environment.

**Verification Steps:**
1. Commit the updated `ci.yml` with verbose flags.
2. Push to GitHub and monitor the Actions run.
3. If it passes, the deployment target fixed it. If it fails, use the verbose logs to pinpoint the exact missing dependency and resolve it in the next iteration.
