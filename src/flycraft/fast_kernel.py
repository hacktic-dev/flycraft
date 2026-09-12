"""Exact native specializations. Original source/library remain the checkpoint ABI."""
import ctypes,hashlib,json,subprocess
from pathlib import Path


def install(brain):
    root=Path(__file__).resolve().parents[2]
    folder=root/'.tools/fast-learning-kernel';folder.mkdir(parents=True,exist_ok=True)
    source=(root/'.tools/learning-kernel/kernel.cpp').read_text()
    needle='   int64_t d=now-last[i];if(d<=0)return;'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'''
   // The common active-neuron case: identical arithmetic and update order,
   // without general elapsed-time/refractory/table-index bookkeeping.
   if(d==1){
     if(refractory[i]>1){--refractory[i];last[i]=now;return;}
     refractory[i]=0;
     const float a=av[1],b=ag[1];
     v[i]=-52.f+(v[i]+52.f)*a+current*(1.f-a)+g[i]*(a-b)/3.f;
     g[i]*=b;last[i]=now;return;
   }
''')
    flags=['-O3','-std=c++17','-shared','-static','-Wl,--export-all-symbols']
    changes=['specialize elapsed-one-step integration']
    cpp=folder/'kernel.cpp';dll=folder/'memory.dll';metadata=folder/'build.json'
    digest=hashlib.sha256(source.encode()).hexdigest()
    record=json.loads(metadata.read_text()) if metadata.exists() else {}
    if not dll.exists() or record.get('source_sha256')!=digest or record.get('binary_sha256')!=hashlib.sha256(dll.read_bytes()).hexdigest():
        cpp.write_text(source)
        compiler=next((root/'.tools').glob('llvm-mingw-*/bin/clang++.exe'))
        subprocess.run([str(compiler),*flags,str(cpp),'-o',str(dll)],check=True)
        record={'source_sha256':digest,'binary_sha256':hashlib.sha256(dll.read_bytes()).hexdigest(),
            'reference_build':brain.build,'flags':flags,'exact':True,'changes':changes}
        metadata.write_text(json.dumps(record,indent=2))
    lib=ctypes.CDLL(str(dll));fn=lib.memory_advance
    fn.argtypes=brain.advance.argtypes;fn.restype=None
    brain.library=lib;brain.advance=fn;brain.runtime_build=record
    # brain.build intentionally continues to identify the reference numerical
    # model for existing checkpoint compatibility. runtime_build identifies the
    # actual executable and is recorded separately in run provenance.
