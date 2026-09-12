"""Short, repeatable full-network speed benchmark. Startup is excluded."""
import argparse,ctypes,json,subprocess,time,sys
from pathlib import Path
import numpy as np
from flycraft.learning import LearningFly,ROOT
from flycraft import r8_visual,learning
from flycraft.game import make_game,initialize_game,pixels,preview_pixels,map_action
from flycraft.training_game import reset_episode,observe
from flycraft.training_metrics import SustainedAttack,EpisodeMotionStats,reward_components
from flycraft.train import panel,motor_panel

OUT=ROOT/'artifacts/speed'

def instrument(brain):
    optimized=hasattr(brain,'runtime_build')
    source=(ROOT/('.tools/fast-learning-kernel/kernel.cpp' if optimized else '.tools/learning-kernel/kernel.cpp')).read_text()
    source='#include <chrono>\nstatic double prof[4]={};\nusing BenchClock=std::chrono::steady_clock;\nstatic double seconds(BenchClock::time_point t){return std::chrono::duration<double>(BenchClock::now()-t).count();}\nextern "C" void read_profile(double* p){for(int i=0;i<4;i++){p[i]=prof[i];prof[i]=0;}}\n'+source
    source=source.replace(' const int delay=', ' auto scan_start=BenchClock::now();\n const int delay=',1)
    source=source.replace(' for(int t=0;t<steps;t++,(*clock)++){',' prof[0]+=seconds(scan_start);\n for(int t=0;t<steps;t++,(*clock)++){\n auto active_start=BenchClock::now();',1)
    source=source.replace('   *nactive=kept;','   *nactive=kept;\n prof[1]+=seconds(active_start);auto edge_start=BenchClock::now();',1)
    source=source.replace('       if(learning_enabled){','       if(learning_enabled){\n auto plastic_start=BenchClock::now();',1)
    source=source.replace('       }\n       continue;', '       prof[3]+=seconds(plastic_start);\n       }\n       continue;',1)
    source=source.replace('   queue_count[slot]=0;', '   prof[2]+=seconds(edge_start);\n   queue_count[slot]=0;',1)
    source=source.replace(' // Materialize all states', ' scan_start=BenchClock::now();\n // Materialize all states',1)
    pos=source.rfind('}');source=source[:pos]+'prof[0]+=seconds(scan_start);\n'+source[pos:]
    cpp=OUT/'profile.cpp';dll=OUT/'profile.dll';cpp.write_text(source)
    compiler=next((ROOT/'.tools').glob('llvm-mingw-*/bin/clang++.exe'))
    flags=brain.runtime_build['flags'] if optimized else ['-O3','-std=c++17','-shared','-static','-Wl,--export-all-symbols']
    subprocess.run([str(compiler),*flags,str(cpp),'-o',str(dll)],check=True)
    lib=ctypes.CDLL(str(dll));fn=lib.memory_advance;fn.argtypes=brain.advance.argtypes;fn.restype=None;brain.advance=fn
    lib.read_profile.argtypes=[ctypes.c_void_p];lib.read_profile.restype=None
    return lib

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='before');ap.add_argument('--live',action='store_true');ap.add_argument('--reference',action='store_true');ap.add_argument('--profile-native',action='store_true');ap.add_argument('--ticks',type=int,default=40);args=ap.parse_args()
    OUT.mkdir(exist_ok=True);c=json.loads((ROOT/'training.json').read_text());g=json.loads((ROOT/'config/baseline.json').read_text());g['port']=8031
    factory=LearningFly
    if args.reference:
        from test_speed_exact import load_reference
        factory=load_reference()
    fly=factory(c['plasticity']);b=fly.brain;lib=instrument(b) if args.profile_native else None
    times={'retina':0.,'native':0.,'brain_total':0.,'minecraft':0.,'logging':0.,'recording':0.,'python_metrics':0.}
    retinal=r8_visual.retinal_samples
    def retinal_timer(*a,**kw):
        t=time.perf_counter();out=retinal(*a,**kw);times['retina']+=time.perf_counter()-t;return out
    r8_visual.retinal_samples=retinal_timer;learning.retinal_samples=retinal_timer
    if args.reference:
        for name in ('flycraft._speed_reference_r8','flycraft._speed_reference_learning'):
            sys.modules[name].retinal_samples=retinal_timer
    attack=SustainedAttack(c['attack']);motion=EpisodeMotionStats();reward=0.;env=None;inputs=[];signals=[];records=[]
    state={'step':0,'episode':500,'history':[{'episode':i+1,'reward':float(i%7),'mean_cumulative_reward':float(i%7)/2} for i in range(500)],'best_eval':0.}
    trace=dict(np.load(OUT/'input-trace.npz')) if not args.live else None
    log=open(OUT/(args.label+'-metrics.jsonl'),'w')
    try:
        if args.live:
            env=make_game(g);initialize_game(env,g);obs=reset_episode(env,{'x':.5,'z':.5,'yaw':35.,'target':[2,-58,5]})
        total_start=time.perf_counter();measured_start=None
        for tick in range(args.ticks+8):
            if tick==8:
                for k in times:times[k]=0.
                if lib:lib.read_profile(np.zeros(4,np.float64).ctypes.data)
                measured_start=time.perf_counter()
            if args.live:rgb=pixels(obs,g);wide=preview_pixels(obs);before=observe(obs,[2,-58,5])
            else:rgb=trace['rgb'][tick]
            inputs.append(rgb.copy())
            t=time.perf_counter();control,neural=fly.step(rgb,record=False);times['brain_total']+=time.perf_counter()-t;times['native']+=neural['neural_compute_seconds']
            if args.live:
                action=map_action(control,g);action['attack']=attack.step(control['attack'])
                t=time.perf_counter();obs=env.step(action)[0];times['minecraft']+=time.perf_counter()-t
                t=time.perf_counter();after=observe(obs,[2,-58,5]);behavior=motion.update(before,after,action)
                signal,parts=reward_components(before,after,c['reward']);reward+=signal;state['step']=tick+1
                p=panel(state,args.ticks,reward,after,parts,c);p.update(episode_step=tick+1,motor=motor_panel(control,action,attack,g),behavior=behavior);fly.training=p
                row={'step':tick+1,'episode':501,'episode_reward':reward,'reward':signal,'components':parts,'state':after,'action':action,'behavior':behavior,**neural}
                if (tick+1)%20==0:row['plasticity']=b.memory()
                times['python_metrics']+=time.perf_counter()-t
                t=time.perf_counter();log.write(json.dumps(row)+'\n');times['logging']+=time.perf_counter()-t
                records.append((wide.copy(),control,action,before,after,p))
            else:signal=float(trace['signals'][tick])
            fly.reinforce(signal);signals.append(signal)
        total=time.perf_counter()-measured_start
        native_parts=np.zeros(4,np.float64)
        if lib:lib.read_profile(native_parts.ctypes.data)
        result={'label':args.label,'ticks':args.ticks,'ms_per_tick':total*1000/args.ticks,'sim_seconds_per_wall_second':args.ticks*.05/total,
            'ms_per_tick_parts':{k:v*1000/args.ticks for k,v in times.items()},'native_profile_ms':dict(zip(('full_scans_and_tables','active_integration','transmission_including_plasticity','plasticity'),(native_parts*1000/args.ticks).tolist())),
            'neurons':b.n,'edges':len(b.weight),'note':'8 warm-up ticks; startup/checkpoints excluded; default no-record training with 500-episode history. Retina timer covers R1-R6 sampling; R8 and other Python work is residual.'}
        result['ms_per_tick_parts']['python_brain_overhead']=max(0,(times['brain_total']-times['native']-times['retina'])*1000/args.ticks)
        if args.live:
            if not (OUT/'input-trace.npz').exists():
                np.savez(OUT/'input-trace.npz',rgb=np.asarray(inputs),signals=np.asarray(signals))
            # Separately price the existing expensive recording path.
            from flycraft.training_recording import EpisodeRecording
            rec=EpisodeRecording(OUT/(args.label+'-recording'),fly,c['recording'],g,records[-1][3],state['history'],preview=False)
            try:
                elapsed=0.
                for i in range(4):
                    wide,control,action,before,after,p=records[-4+i];rgb=inputs[-4+i]
                    control,neural=fly.step(rgb,record=True);fly.training=p
                    t=time.perf_counter();rec.write(fly,rgb,wide,control,action,before,after,i,neural,p);elapsed+=time.perf_counter()-t
                result['recording_extra_ms_per_tick']=elapsed*250
            finally:rec.close({'benchmark':True})
        (OUT/(args.label+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    finally:
        log.close()
        if env:env.close()

if __name__=='__main__':main()
