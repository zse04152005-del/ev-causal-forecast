"""Conditional energy accounting; see round3_figure_contract.md and transfer_meta.json."""
from pathlib import Path
import pandas as pd
from figlib import plt,setup_rc,PAL,MM
root=Path(__file__).resolve().parents[2];setup_rc();d=pd.read_csv(root/'results/scenarios/transfer/transfer_profiles.csv')
fig,axes=plt.subplots(1,2,figsize=(190*MM,65*MM),sharey=True)
for ax,(e,title) in zip(axes,[('anchor','(a) Window elasticity: −0.21'),('arrival_mu60','(b) Arrival elasticity: −0.31')]):
    for mode,label,color,ls in [('disappear','Energy disappears',PAL['gray']['stroke'],'--'),('shift','Shift to low-price hours',PAL['causal']['stroke'],'-')]:
        q=d[(d.elasticity==e)&(d['mode']==mode)].sort_values('hour');v=q.delta_MW.to_numpy()
        ax.step(range(25),[*v,v[-1]],where='post',color=color,ls=ls,lw=1.2,label=label)
    ax.axhline(0,color='#999999',lw=.5);ax.axvspan(18,22,color='#eeeeee',zorder=0)
    ax.set(xlim=(0,24),xticks=[0,6,12,18,24],xlabel='Hour of day',title=title)
    ax.spines[['right','top']].set_visible(False);ax.legend(frameon=False,fontsize=7,loc='upper left')
axes[0].set_ylabel('Change in mean charging load (MW)');fig.tight_layout(w_pad=1.6)
for ext in ['pdf','svg','png']:fig.savefig(root/f'figures/results/Fig_S_transfer.{ext}',dpi=600)
