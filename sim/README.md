# sim/ - golden model, tools, RTL, testbenches and tests

Everything here runs from this folder:

```
python run_all_tests.py
```

| Kind | Files |
|---|---|
| Tools and golden model | `assembler.py`, `simulator.py`, `gpu_model.py`, `sfu_check.py` |
| RTL (design) | `cu.v`, `tex_unit.v`, `rasterizer.v`, `prim_unit.v`, `rop.v`, `display_engine.v`, `gpu_core.v`, `gpu_top.v` |
| Testbenches | `tb_*.v` |
| Test runners | `run_*.py` (one per block) and `run_all_tests.py` |
