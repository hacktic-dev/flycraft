FlyCraft R8 runtime + dashboard fix

Changes:
- fixes R8VisualMemoryBrain so DOOMFLY-v6 per-receptor R8 current arrays are accepted by the v1 neural step (fixes TypeError: only size-1 arrays can be converted to Python scalars)
- keeps scalar DAN/PPL101 stimulation working
- retina dashboard now draws R1-R6 as gray/white, R8p as blue, R8y as green
- verify_r8_input.py now actually executes an R8 rgb_step smoke test so the vector-current bug is caught before starting Minecraft

After copying the patch into the project root, run:
  . .\env.ps1
  & $python scripts\verify_r8_input.py

Then rerun:
  & $python scripts\diagnose_kc_live.py
