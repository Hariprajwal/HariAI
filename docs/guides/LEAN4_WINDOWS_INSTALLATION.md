# Lean 4 Installation & Setup Guide for Windows

This guide provides the official, verified instructions for installing **Lean 4** and its toolchain manager **`elan`** on Windows 10/11 for use with HariAI's Formal Verification Engine.

---

## 1. Quick PowerShell Installation (Recommended)

1. Open a standard **PowerShell** terminal (run as your current user).
2. Download the official installer script from `lean-lang.org`:
   ```powershell
   curl -O --location https://elan.lean-lang.org/elan-init.ps1
   ```
3. Run the installer script with bypass execution policy:
   ```powershell
   powershell -ExecutionPolicy Bypass -f elan-init.ps1
   ```
4. Follow the interactive prompts (press Enter to accept standard defaults). It will install:
   - `elan` (the version manager)
   - `lean` (the Lean 4 compiler and theorem prover)
   - `lake` (Lean's package manager and build system)
   - Automatically adds `~/.elan/bin` to your User `PATH`.
5. Clean up the installer script:
   ```powershell
   del elan-init.ps1
   ```
6. **Restart your PowerShell terminal** so the updated `PATH` environment variable takes effect.

---

## 2. Verify Your Installation

In a new PowerShell window, run:
```powershell
elan --version
lean --version
lake --version
```
Expected output:
```text
elan 3.x.x ...
Lean (version 4.x.x, ...)
Lake (version 5.x.x, ...)
```

---

## 3. Verify a Lean 4 Theorem from the Command Line

Create a quick test file `test_math.lean`:
```lean
theorem add_comm_test (a b : Nat) : a + b = b + a := by
  omega

#eval 1 + 1
```

Run it directly:
```powershell
lean --run test_math.lean
```
If output displays `2` with exit code `0`, **Lean 4 is fully operational on your machine!**

---

## 4. How HariAI Integrates with Lean 4

HariAI's `hari_ai/verification/lean_kernel.py` automatically checks if `lean` is present in your system `PATH`:
- **If Lean 4 is installed:** HariAI compiles mathematical theorems and logic proofs through the trusted Lean kernel mechanically. Proofs that pass receive $+1.0$ verifiable rewards. Proofs that fail return compiler diagnostics to the **Dream Engine** for sleep-time correction.
- **If Lean 4 is not yet installed:** HariAI automatically falls back to deterministic Python assertion sandboxing (`exec(code + tests)`).

