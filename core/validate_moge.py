"""Validate the pretrained local MoGe implementation on an image (one pipeline stage).

uv run --no-project --env-file .env.macos .venv/bin/python -m genia.core.validate_moge \
    --image demo/data/image/generated_elephant/0000.png --output results/macos-port/moge
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from genia.core.device import get_device, synchronize
from moge.model.v1 import MoGeModel


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tokens',type=int,default=2500)
    ap.add_argument('--compare-cpu',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(4)
    device=get_device()
    args.output.mkdir(parents=True,exist_ok=True)
    rgb=np.array(Image.open(args.image).convert('RGB'))
    image=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
    start=time.monotonic()
    model=MoGeModel.from_pretrained('Ruicheng/moge-vitl',local_files_only=True).eval().to(device)
    synchronize(device)
    stats={'stage':'MoGe depth only; full GenIA reconstruction is separate',
           'model':'Ruicheng/moge-vitl','image':str(args.image),'device':str(device),
           'tokens':args.tokens,'load_seconds':time.monotonic()-start}
    print('Loaded pretrained MoGe',flush=True)
    start=time.monotonic()
    result=model.infer(image,num_tokens=args.tokens)
    synchronize(device)
    stats['inference_seconds']=time.monotonic()-start
    data={k:v.cpu().numpy() for k,v in result.items()}
    valid=data['mask'] & np.isfinite(data['depth']) & (data['depth']>0)
    stats['valid_fraction']=float(valid.mean())
    if not valid.any(): raise RuntimeError('MoGe returned no valid positive depth')
    stats['valid_depth_quantiles']=np.quantile(data['depth'][valid],[.01,.5,.99]).tolist()
    stats['intrinsics']=data['intrinsics'].tolist()
    np.savez_compressed(args.output/'prediction.npz',**data)
    import trimesh
    trimesh.points.PointCloud(data['points'][valid][::8],colors=rgb[valid][::8]).export(args.output/'points.ply')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(rgb); ax[0].set_title('Generated input')
    lo,hi=np.quantile(data['depth'][valid],[.02,.98])
    im=ax[1].imshow(np.ma.array(data['depth'],mask=~valid),cmap='viridis',vmin=lo,vmax=hi)
    ax[1].set_title('Pretrained MoGe depth (MPS)')
    for a in ax:a.axis('off')
    fig.colorbar(im,ax=ax[1],fraction=.046,pad=.04)
    fig.savefig(args.output/'depth.png',dpi=150,bbox_inches='tight');plt.close(fig)
    if args.compare_cpu:
        # Compare fp32 predictions before camera fitting at a smaller token count,
        # using the same pretrained weights and full-resolution input.
        tokens=min(args.tokens,256)
        with torch.inference_mode():
            metal=model(image[None].to(device),num_tokens=tokens)
            metal={k:v.cpu() for k,v in metal.items()}
            model=model.cpu(); torch.mps.empty_cache() if device.type=='mps' else None
            reference=model(image[None],num_tokens=tokens)
        stats['cpu_comparison_tokens']=tokens
        stats['cpu_max_abs_errors']={k:float((metal[k]-reference[k]).abs().max()) for k in reference}
        for key in reference:
            torch.testing.assert_close(metal[key],reference[key],atol=5e-4,rtol=3e-3)
    (args.output/'validation.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__': main()
