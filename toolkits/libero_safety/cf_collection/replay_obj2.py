import json, os, sys, numpy as np
from rlinf.envs.sim.libero.utils import get_benchmark_overridden
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
p=sys.argv[1]; j=json.load(open(p)); fr=np.load(os.path.join(os.path.dirname(p),j["visual_trajectory"]["path"]))["agentview_image"]
ep=j["episode"]; suite=get_benchmark_overridden(ep["task_suite"])(); task=suite.get_task(ep["task_id"])
env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task.level,task.level_id),camera_heights=256,camera_widths=256,ignore_done=True); env.seed(0); env.reset()
obs=env.set_init_state(suite.get_task_init_states(task.level,task.level_id)[ep["trial_id"]])
for _ in range(15): obs,*_=env.step([0,0,0,0,0,0,-1])
dom=env.env
for name,gen in dom.mocap_motion_generators.items():
    rec=np.array(j["initial_observation"][name+"_pos"]); cur=np.array(obs[name+"_pos"]); off=rec-cur
    print("align",name,"offset",off, "step_count",gen.step_count, "half",gen.half_steps)
    gen.pos=gen.pos+off; gen.start_pos=gen.start_pos+off
    dom.sim.data.set_mocap_pos(name+"_main_mocap", dom.sim.data.get_mocap_pos(name+"_main_mocap")+off)
dom.sim.forward()
keys=["robot0_eef_pos","toy_train_1_pos","white_yellow_mug_1_pos","robot0_gripper_qpos"]
io=j["initial_observation"]; print("init", {k: float(np.abs(np.array(obs[k])-np.array(io[k])).max()) for k in keys})
Image.fromarray(np.concatenate([obs["agentview_image"][::-1,::-1],fr[0],np.abs(obs["agentview_image"][::-1,::-1].astype(int)-fr[0].astype(int)).clip(0,255).astype(np.uint8)*4],1)).save("/mnt/nfs/fjj/Code/rlinf-safety-runtime/tmp/replay_out/f0_cmp.png")
for t,s in enumerate(j["trajectory"]):
    obs,*_=env.step(s["action"]); o=s["observation"]; dimg=np.abs(obs["agentview_image"][::-1,::-1].astype(int)-fr[t+1].astype(int)).mean()
    d={k: float(np.abs(np.array(obs[k])-np.array(o[k])).max()) for k in keys}
    if t%50==0 or t==len(j["trajectory"])-1: print(t, f"img{dimg:.2f}", {k:f"{v:.1e}" for k,v in d.items()})
    
