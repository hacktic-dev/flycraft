"""Build the unchanged upstream kernel as a Windows DLL with its hash guard."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'vendor/doomfly/doom/kernel.cpp'
output=ROOT/'vendor/doomfly/outputs/doom/libneural.so'
output.parent.mkdir(parents=True,exist_ok=True)
compiler=next((ROOT/'.tools').glob('llvm-mingw-*/bin/clang++.exe'))
temporary=output.with_suffix('.partial.dll')
command=[str(compiler),'-O3','-std=c++17','-shared','-static','-Wl,--export-all-symbols',str(source),'-o',str(temporary)]
subprocess.run(command,check=True)
temporary.replace(output)
record={'kernel_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'model_revision':'lif-r2-refractory-write-protection','compile_flags':command[1:-3],'platform':'Windows x64 PE DLL with upstream-compatible filename'}
output.with_suffix(output.suffix+'.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
