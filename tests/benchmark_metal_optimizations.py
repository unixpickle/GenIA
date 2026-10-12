"""Reproduce old/new Metal timings: uv run python tests/benchmark_metal_optimizations.py.

Uses synthetic activations, synchronized wall time, one warm-up and three samples.
Attention uses the full ActionMesh shape; this is not an end-to-end model benchmark.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch
from genia.core.ops.attention import scaled_dot_product_attention
from genia.core.ops.gaussian import _MetalSplat, gaussian_tile_intersections
from genia.core.ops.knn import knn_points
from genia.core.ops.sparse import submanifold_conv3d, _submanifold_conv3d_reference


def reference(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / "reference" / (name + ".py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def compare(name, old_fn, new_fn, inputs, gradient=False, autocast=False):
    records, saved = {}, {}
    for label, fn in [("baseline", old_fn), ("optimized", new_fn)]:
        args = [x.detach().requires_grad_(gradient) for x in inputs]
        forward, backward = [], []
        for iteration in range(4):
            for x in args: x.grad = None
            torch.mps.synchronize(); start = time.perf_counter()
            with torch.set_grad_enabled(gradient), torch.autocast("mps", dtype=torch.bfloat16, enabled=autocast):
                output = fn(*args)
            values = (output,) if isinstance(output, torch.Tensor) else tuple(output)
            torch.mps.synchronize(); middle = time.perf_counter()
            if gradient:
                torch.autograd.backward(values, [torch.ones_like(x) for x in values])
                torch.mps.synchronize()
            end = time.perf_counter()
            if iteration:
                forward.append(middle - start); backward.append(end - middle)
        saved[label] = [x.detach().float().cpu() for x in values]
        if gradient: saved[label] += [x.grad.detach().float().cpu() for x in args]
        records[label] = {"forward_seconds": statistics.median(forward),
                          "backward_seconds": statistics.median(backward) if gradient else None,
                          "samples_forward": forward, "samples_backward": backward if gradient else None}
        del output, values, args
        gc.collect(); torch.mps.empty_cache()
        print(name, label, json.dumps(records[label]), flush=True)
    errors = [{"max_absolute": float((a-b).abs().max()) if a.numel() else 0,
               "relative_l2": float((a-b).norm()/b.norm().clamp_min(1e-20))}
              for a, b in zip(saved["optimized"], saved["baseline"])]
    assert all(x["relative_l2"] < (.012 if autocast else .002) for x in errors), errors
    report = {"name": name, "gradient": gradient, "autocast_bfloat16": autocast,
              "input_shapes": [list(x.shape) for x in inputs], "runs": records,
              "errors_outputs_then_gradients": errors,
              "forward_speedup": records["baseline"]["forward_seconds"] / records["optimized"]["forward_seconds"]}
    if gradient:
        report["backward_speedup"] = records["baseline"]["backward_seconds"] / records["optimized"]["backward_seconds"]
        report["total_speedup"] = sum(records["baseline"][k] for k in ["forward_seconds", "backward_seconds"]) / sum(records["optimized"][k] for k in ["forward_seconds", "backward_seconds"])
    print(name, 'comparison', json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/macos-port/kernel_optimization/validation.json"))
    parser.add_argument("--skip-attention", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(911); torch.set_num_threads(2)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(record):
        records.append(record)
        args.output.write_text(json.dumps({"torch": torch.__version__, "device": "mps",
            "scope": "Synthetic operator benchmarks; one warm-up, median of three synchronized runs. Not CUDA or end-to-end timings.",
            "records": records}, indent=2) + "\n")
    if not args.skip_attention:
        q = torch.randn(2, 32784, 16, 128, device="mps").transpose(1, 2)
        k = torch.randn_like(q)
        v = torch.randn(2, 32784, 16, 384, device="mps", dtype=torch.bfloat16).split(128, dim=-1)[2].transpose(1, 2)
        save(compare("actionmesh_attention", lambda q,k,v: scaled_dot_product_attention(q,k,v,max_score_elements=16*1024**2),
                     scaled_dot_product_attention, [q,k,v], autocast=True))
        del q,k,v; gc.collect(); torch.mps.empty_cache()
    old_knn = reference("knn_baseline").knn_points
    for grad in [False, True]:
        q,k=[torch.randn(1,8192,3,device="mps") for _ in range(2)]
        save(compare("knn_8192" + ("_training" if grad else ""), lambda q,k: old_knn(q,k).dists,
                     lambda q,k: knn_points(q,k).dists,[q,k],gradient=grad))
    del q,k
    for n,c in [(4096,64),(19650,128),(19650,512)]:
        features=torch.randn(n,c,device="mps");weight=torch.randn(c,3,3,3,c,device="mps")*.01
        neighbors=torch.randint(-n//2,n,(27,n),device="mps").clamp_min(-1)
        save(compare(f"sparse_{n}_{c}",lambda f,w: _submanifold_conv3d_reference(f,w,neighbors),
                     lambda f,w: submanifold_conv3d(f,w,neighbors),[features,weight],gradient=True))
        del features,weight,neighbors;gc.collect();torch.mps.empty_cache()
    old_splat=reference("splat_baseline")._MetalSplat.apply
    for n,r in [(1000,8),(10000,8),(1000,80)]:
        h=w=256
        means=torch.rand(n,2,device="mps")*w
        conics=torch.tensor([1/r**2,0,1/r**2],device="mps").expand(n,3).contiguous()
        inputs=[means,conics,torch.rand(n,4,device="mps"),torch.rand(n,device="mps")*.1,torch.zeros(4,device="mps")]
        radii=torch.full((n,2),r*3,device="mps",dtype=torch.int32)
        ranges,ids=gaussian_tile_intersections(means,radii,torch.rand(n,device="mps"),w,h)
        save(compare(f"splat_{n}_radius{r}", lambda *x: old_splat(*x,ranges,ids,w,h,16),
                     lambda *x: _MetalSplat.apply(*x,ranges,ids,w,h,16),inputs,gradient=True))


if __name__ == "__main__": main()
