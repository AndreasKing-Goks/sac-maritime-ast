# SAC-Maritime-AST

Repository for **Soft Actor-Critic Adaptive Stress Testing (SAC-AST)** framework in maritime applications.

---

## Environment Setup

### TensorFlow-GPU Compatibility

Before installation, ensure compatibility between **TensorFlow**, **CUDA**, and **cuDNN**:

- [TensorFlow GPU compatibility guide](https://www.tensorflow.org/install/source#gpu)
- [Windows native pip install instructions](https://www.tensorflow.org/install/pip#windows-native)

> **Note:** This project runs on a local Windows-native machine.

### TensorFlow Dependencies

All required packages are listed in [`sac-test-gpu.yml`](sac-test-gpu.yml).

- Downgrade `numpy` to **<2.0** due to known compatibility issues with TensorFlow.

To verify `tensorflow` with GPU support installation:

```bash
python test_beds/test_gpu.py
```

## Installing `rllab` Package

To run the SAC implementation, install the `rllab` dependency from Haarnoja’s original SAC repository:  
[https://github.com/haarnoja/sac/tree/master](https://github.com/haarnoja/sac/tree/master)

---

### For Windows PowerShell

Add the `rllab` directory to your `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Get-Location).Path + ";" + $env:PYTHONPATH
```

### For Visual Studio Code Users

Ensure VS Code recognizes the local `rllab` package by updating your workspace settings.

Edit or create the file `.vscode/settings.json` in your project root and add the following:

```json
{
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "D:/OneDrive - NTNU/PhD/PhD_Projects/sac-maritime-ast/rllab"
  },
  "python.analysis.extraPaths": [
    "rllab"
  ]
}
```

To verify `rllab` package installation:

```bash
python test_beds/test_rllab.py
```
