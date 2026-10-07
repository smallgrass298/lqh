"""Read full-run logs, audit completeness, plot current-version diagnostics."""
from pathlib import Path
import os,re,json,statistics as st
R=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(R/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=R;out=R/'results/generated/current_analysis';out.mkdir(parents=True,exist_ok=True)
keys=['plan','pack','forward','remote','backward','unpack']
data={};audit=[];same_traffic={}
for g in (1,2,4,8):
 d=next((root/f'output_{g}/time_record/raw').glob(f'*_{g}gpu_suite_full'));mode='dynamic' if g==1 else 'dynamic_all';records=[]
 assert (d/'correctness.txt').read_text().startswith('PASS:')
 for m in (['sorted','dynamic'] if g==1 else ['all','dynamic_all']):
  for rep in (1,2,3):
   log=(d/f'gpu_{m}_rep{rep}.log').read_text()
   assert int(re.findall(r'GPU Iteration: (\d+)',log)[-1])==2834
   assert (d/f'check_{m}_rep{rep}.txt').read_text().startswith('PASS:')
   if m!=mode:continue
   r={'rep':rep,'loop':float(re.findall(r'GPU Loop time:\s*(\S+)',log)[-1])}
   sched=re.findall(r'GPU reaction scheduler rank=(\d+) mode=(\w+) dynamic_launches=(\d+) bucket_evaluations=(\d+) bucket_bypasses=(\d+) worker_blocks=(\d+) chunk=(\d+)',log)
   assert len(sched)==g and {int(v[0]) for v in sched}==set(range(g))
   r['ranks']={int(v[0]):dict(zip(['launches','evaluations','bypasses','blocks','chunk'],map(int,v[2:]))) for v in sched}
   if g>1:
    line=re.search(r'GPU DLB timing\(max-rank cumulative s\): (.*)',log)[1]
    r['phases']={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',line)}
    line=re.search(r'GPU DLB traffic\(global cumulative\): (.*)',log)[1]
    r['traffic']={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
    old=(d/f'gpu_all_rep{rep}.log').read_text()
    oldline=re.search(r'GPU DLB traffic\(global cumulative\): (.*)',old)[1]
    assert r['traffic']=={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',oldline)}
    same_traffic[g]=True
   records.append(r)
 data[g]=records
 audit.append(f'{g} GPU: all 6 full logs reached step 2834; all 6 saved comparison checks PASS; per-rank records complete.')
(out/'parsed_current_logs.json').write_text(json.dumps(data,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
colors=['#608DB1','#E8963C','#37A36C']
def save(fig,name):
 fig.savefig(out/(name+'.png'),dpi=210);fig.savefig(out/(name+'.pdf'));plt.close(fig)
gs=[2,4,8]
fig,ax=plt.subplots(figsize=(12,6.4));fig.subplots_adjust(left=.085,right=.98,top=.79,bottom=.26)
fig.suptitle('Dynamic Version: DLB Phase Timings',fontsize=16,weight='bold',y=.96)
fig.text(.5,.895,'281×141×32, 2834 steps | Mean of 3 full runs',ha='center',fontsize=11)
x=np.arange(6);w=.24
for j,g in enumerate(gs):
 vals=[st.mean(r['phases'][k] for r in data[g]) for k in keys];sd=[st.stdev(r['phases'][k] for r in data[g]) for k in keys]
 bars=ax.bar(x+(j-1)*w,vals,w,color=colors[j],label=f'{g} GPUs',yerr=sd,capsize=2)
 for b,v in zip(bars,vals):ax.text(b.get_x()+w/2,v+2.5,f'{v:.1f}' if v>=.1 else f'{v:.3f}',ha='center',fontsize=8)
ax.set_xticks(x,['Planning','Pack + sync','Forward MPI','Remote stage','Return MPI','Unpack launch']);ax.set_ylabel('Cumulative host-stage time (s)');ax.grid(axis='y',ls='--',alpha=.3);ax.legend(ncol=3);ax.set_ylim(0,ax.get_ylim()[1]*1.16)
fig.text(.085,.14,'Each phase is the maximum cumulative time across ranks; error bars show sample SD across 3 runs.',fontsize=10)
fig.text(.085,.10,'Phases overlap. MPI timers include waiting. Remote stage includes local-work launch and remote-stream synchronisation.',fontsize=9)
fig.text(.085,.06,'Unpack records host submission time. Local reaction kernel time was not recorded separately.',fontsize=9,color='#555555')
save(fig,'01_phase_timings')
fig,axs=plt.subplots(1,2,figsize=(12,6.3));fig.subplots_adjust(left=.08,right=.98,top=.79,bottom=.21,wspace=.3)
fig.suptitle('Dynamic Version: Reaction-Task MPI Traffic',fontsize=16,weight='bold',y=.96)
fig.text(.5,.895,'Full-run DLB transfers, summed across ranks | Identical counts in all 3 repeats',ha='center',fontsize=11)
x=np.arange(3);w=.34
for j,(key,label,c) in enumerate([('forward','Task inputs',colors[0]),('backward','Returned results',colors[2])]):
 volume=[data[g][0]['traffic'][key+'_bytes']/1024**3 for g in gs]
 messages=[data[g][0]['traffic'][key+'_messages'] for g in gs]
 bars=axs[0].bar(x+(j-.5)*w,volume,w,label=label,color=c)
 for b,v in zip(bars,volume):axs[0].text(b.get_x()+w/2,v+.015,f'{v:.3f}',ha='center',fontsize=10)
 sizes=[data[g][0]['traffic'][key+'_bytes']/n/1024 for g,n in zip(gs,messages)]
 bars=axs[1].bar(x+(j-.5)*w,sizes,w,label=label,color=c)
 for b,v in zip(bars,sizes):axs[1].text(b.get_x()+w/2,v+.4,f'{v:.1f}',ha='center',fontsize=10)
axs[0].set_ylabel('Transferred data (GiB)');axs[1].set_ylabel('Mean message size (KiB)')
for ax in axs:ax.set_xticks(x,[f'{g} GPUs' for g in gs]);ax.set_ylim(0,ax.get_ylim()[1]*1.25);ax.grid(axis='y',ls='--',alpha=.3);ax.legend(fontsize=10)
fig.text(.08,.105,'Counts cover DLB task migration. Halo exchanges and planning collectives are outside these traffic counters.',fontsize=10)
fig.text(.08,.065,'1 GPU bypasses cross-rank DLB. Message size is total sent bytes divided by total send-message count.',fontsize=9,color='#555555')
save(fig,'02_traffic')
fig,axs=plt.subplots(2,2,figsize=(12,8));fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.17,hspace=.50,wspace=.24)
fig.suptitle('Dynamic Queue Activity by MPI Rank',fontsize=16,weight='bold',y=.96)
fig.text(.5,.90,'Number of timesteps using the dynamic local-reaction kernel (2834 total)',ha='center',fontsize=11)
for ax,g in zip(axs.flat,[1,2,4,8]):
 values=[data[g][0]['ranks'][rank]['launches'] for rank in range(g)]
 assert all([r['ranks'][rank]['launches'] for rank in range(g)]==values for r in data[g])
 bars=ax.bar(np.arange(g),values,color=colors[2],width=.58)
 for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+60,str(v),ha='center',fontsize=10)
 ax.set_title(f'{g} GPU'+('s' if g>1 else ''));ax.set_xticks(range(g));ax.set_xlabel('MPI rank');ax.set_ylim(0,3100);ax.grid(axis='y',ls='--',alpha=.3)
axs[0,0].set_ylabel('Dynamic launches');axs[1,0].set_ylabel('Dynamic launches')
fig.text(.08,.085,'Counts match across all 3 repeats. Zero means the local solver used the regular kernel on every timestep.',fontsize=10)
fig.text(.08,.05,'The bypass uses the fraction of unmasked cells with Nchem > 1. These counters measure activation, not warp balance.',fontsize=9,color='#555555')
save(fig,'03_rank_activity')
lines=['# 动态版日志分析','',*audit,'','## 当前测量结果','']
for g in (1,2,4,8):
 rs=data[g];lines.append(f'- {g} GPU：loop {st.mean(r["loop"] for r in rs):.3f} s；逐 rank 动态启用次数 {[rs[0]["ranks"][i]["launches"] for i in range(g)]}。')
 if g>1:
  ph={k:st.mean(r['phases'][k] for r in rs) for k in keys};tr=rs[0]['traffic']
  lines.append(f'  - 主机阶段计时（各 rank 累计值取最大后，3 次均值）：{ph}。')
  lines.append(f'  - 正向 {tr["forward_bytes"]/1024**3:.4f} GiB，返回 {tr["backward_bytes"]/1024**3:.4f} GiB；单方向 {tr["forward_messages"]} 条消息。')
lines+=['','## 能支持的判断','',
'1. 2/4/8 GPU 三次运行中，动态版与同批分桶版的 DLB 输入/结果字节数及消息数逐项相同。新增动态领取没有增加这些已记录的 MPI 数据量。总 MPI 流量还包括未计入的 halo 和 collectives。',
'2. 8 GPU 的 rank 0–3 确实执行了动态 kernel，rank 4–7 全程旁路。此前用 MPI_MAX 后的 bypass=142 推断所有 rank 都旁路是不成立的。',
'3. 动态启用次数的空间差异说明各 rank 触发调度的情况不同，不能直接量化 rank 工作量差或 warp 内不平衡。旁路也不代表没有计算、没有远端任务或 Nchem 全部等于 1。',
'4. 已有单 GPU 对照支持 GPU 内部执行路径带来了净收益；目前无法将收益分解成求解计算、访存、任务分配或调度开销各自的贡献。',
'','## 计时边界','',
'forward/return 是 post+wait 的主机耗时，含等待；remote 包含远端 kernel 提交、本地 launch_local_reaction 调用及远端 stream 同步；pack 包含同步；unpack 主要是异步提交耗时。各阶段可重叠，最大值还可能来自不同 rank，不能求和或用 loop 减去它们算纯计算。',
'','## 当前缺失的诊断','',
'- 每 rank 本地/远端反应 kernel 的 GPU 时间，以及同一时间步内的重叠和暴露等待，需要 CUDA event 或时间线追踪。',
'- 当前实际任务列表中的 Nchem 分布，尤其按动态队列的实际 32 格点任务块计算的迭代离散度。历史 spatial mapping 的 warp waste 不适用于当前队列。',
'- 每 warp 领取块数、累计工作量、完成时间分布及最后阶段拖尾；领取块数不等本身可能是动态均衡的正常结果。',
'- 可用 1 - sum(nc)/(active_lanes * max(nc)) 作为块内迭代不均衡代理，尾块 padding 单独计数。该代理不是实测周期、硬件 warp efficiency 或真实反应耗时。',
'- 远端 kSolveReactionTasks 仍采用列表处理，也需要独立检查接收任务的分组和块内差异。',
'','本批图只展示动态版；没有重用旧版 Nchem 直方图或虚构 GPU kernel/warp 指标。新诊断应作为独立插桩实验，正式性能继续引用现有 clean runs。',
'','数值输出文件未包含在日志包中；正确性依据已保存的逐次 PASS 记录，本次未重新比较 .dat。']
(out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:20]))
