These two modules freeze the pre-optimisation FlyCraft learning adapter and R8
input processing for bitwise regression and replay benchmarking. They use the
same pinned DOOMFLY graph and original observed kernel as production, but do not
install the accelerated kernel or reuse retinal samples. Do not update these
fixtures to match an optimisation: they are the independent reference.

The test checks every neural state array, every retained weight, spike bins,
controller output, actual aversive depression, frozen evaluation, and checkpoint
loading in both directions. It uses the locally captured benchmark trace when
available, otherwise seeded synthetic RGB frames. Direct KC stimulation is
restricted to this regression assay.
