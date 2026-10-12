"""Check pretrained local MapAnything on real views and compare CPU/Metal numerics.

This validates the depth/camera stage separately from full GenIA reconstruction.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from genia.core.device import get_device, synchronize, empty_cache
from genia.core.utils.map_anything_depth import run_map_anything
from genia.core.utils.model_cache import ModelCache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare-cpu',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=get_device()
    images=[np.array(Image.open(path).convert('RGB')) for path in args.images]
    start=time.monotonic()
    model=ModelCache.get().map_anything_model
    synchronize(device)
    stats={'stage':'MapAnything depth/cameras; full reconstruction is separate',
           'model':ModelCache.get().map_anything_model_id,'device':str(device),
           'images':[str(path) for path in args.images],
           'load_seconds':time.monotonic()-start}
    print('Loaded pretrained MapAnything',flush=True)
    start=time.monotonic()
    results=run_map_anything(images,list(range(len(images))))
    synchronize(device)
    stats['inference_seconds']=time.monotonic()-start
    stats['views']=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(images),2,figsize=(10,5*len(images)),squeeze=False)
    for index,result in results.items():
        depth=result.depth_map_z
        valid=result.valid_mask & np.isfinite(depth) & (depth>0)
        if not valid.any():raise RuntimeError(f'No valid positive depth in view {index}')
        assert np.isfinite(result.K_matrix).all() and np.isfinite(result.c2w).all()
        stats['views'].append({'shape':list(depth.shape),'valid_fraction':float(valid.mean()),
            'depth_quantiles':np.quantile(depth[valid],[.01,.5,.99]).tolist(),
            'intrinsics':result.K_matrix.tolist(),'camera_to_world':result.c2w.tolist()})
        np.savez_compressed(args.output/f'view_{index}.npz',depth=depth,valid=valid,
                            intrinsics=result.K_matrix,camera_to_world=result.c2w,
                            image=result.cropped_image)
        axes[index,0].imshow(result.cropped_image);axes[index,0].set_title(f'View {index}')
        lo,hi=np.quantile(depth[valid],[.02,.98])
        plot=axes[index,1].imshow(np.ma.array(depth,mask=~valid),cmap='viridis',vmin=lo,vmax=hi)
        axes[index,1].set_title('Pretrained MapAnything depth')
        fig.colorbar(plot,ax=axes[index,1],fraction=.046,pad=.04)
        for axis in axes[index]:axis.axis('off')
    fig.savefig(args.output/'depth.png',dpi=150,bbox_inches='tight');plt.close(fig)
    if args.compare_cpu:
        # Small real-image views isolate fp32 model numerics from mask thresholds
        # and camera/depth postprocessing. Full-resolution inference ran above.
        mean=torch.tensor([.485,.456,.406])[:,None,None]
        std=torch.tensor([.229,.224,.225])[:,None,None]
        views=[]
        for rgb in images:
            resized=np.array(Image.fromarray(rgb).resize((112,112),Image.Resampling.BICUBIC))
            tensor=torch.from_numpy(resized).permute(2,0,1).float()/255
            views.append({'img':((tensor-mean)/std)[None],'data_norm_type':['dinov2']})
        with torch.inference_mode():
            actual=model([{**view,'img':view['img'].to(device)} for view in views])
            actual=[{key:value.cpu() for key,value in view.items()} for view in actual]
            model.cpu();empty_cache(device)
            expected=model(views)
        comparisons=[]
        for got,reference in zip(actual,expected):
            errors={}
            for key,value in reference.items():
                assert torch.isfinite(value).all() and torch.isfinite(got[key]).all()
                if value.dtype == torch.bool:
                    agreement=(got[key] == value).float().mean().item()
                    errors[key]={'agreement':agreement}
                    if agreement < .999:
                        raise AssertionError(f'{key}: CPU/Metal mask agreement {agreement}')
                    continue
                delta=got[key].float()-value.float()
                errors[key]={'max_abs':delta.abs().max().item(),
                            'relative_l2':(delta.norm()/value.float().norm().clamp_min(1e-8)).item()}
                torch.testing.assert_close(got[key],value,atol=2e-3,rtol=5e-3)
            comparisons.append(errors)
        stats['cpu_comparison']={'resolution':[112,112],'views':comparisons}
    (args.output/'validation.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
