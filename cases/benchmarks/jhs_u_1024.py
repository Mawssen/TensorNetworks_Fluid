#!/usr/bin/env python3
"""Full-size JHS u cross-check. No submission or scientific tolerance selection."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
DNS = Path("/ix/pgivi/moe32/Schmidt/Forced_Isotropic/5_Grid_1024/Forced_Isotropic_1024Cubed_mat_in_one/U_t1.mat")
DNS_V = DNS.with_name("V_t1.mat")
DNS_W = DNS.with_name("W_t1.mat")
MPS = ROOT / "artifacts/runs/mps/jhs_u_t1/1024cubed/interleaved/cutoff_1e-4/submission-XIcEpK/result/truncated_U_t1_1024_il_cf0.0001.mat"
SPACING = 0.0061359406  # literal 2FI scalar-case spacing, not a new convention
REFERENCES = {"derivative": ROOT/"legacy/mps_py_codes/calculate_spectral_derivative.py",
              "metrics": ROOT/"legacy/mps_py_codes/calculate_u_comp_fidelity_l2norm.py",
              "spectrum": ROOT/"legacy/mps_py_codes/calculate_spectrum.py",
              "float32": ROOT/"legacy/py/jhs_velocity_spectrum_stats.py"}


def reference(path, function=None):
    if function:
        import numpy as np
        tree = ast.parse(path.read_text())
        definitions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==function]
        if len(definitions)!=1:
            raise RuntimeError("missing reference function: "+str(path))
        scope={"np":np,"gc":gc}
        exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),"exec"),scope)
        return scope[function]
    spec=importlib.util.spec_from_file_location("reference_"+path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_field(name):
    import h5py
    import numpy as np
    path,key={"dns":(DNS,"U1"),"dns_v":(DNS_V,"V1"),
              "dns_w":(DNS_W,"W1"),"mps":(MPS,"u")}[name]
    with h5py.File(path,"r") as stream:
        result=np.asarray(stream[key],dtype=np.float64)
    # Exact axis reversal used by BOTH historical Python JHS loaders.
    result=result.transpose(2,1,0)
    if result.shape!=(1024,1024,1024):
        raise RuntimeError("unexpected field shape")
    return result


def worker(args):
    import numpy as np
    import h5py
    sys.path.insert(0,str(ROOT/"src/python"))
    from turbulence_tools_stats import (calculate_spectrum_binned,
        spectral_derivative_slabbed,compare_fields_chunked)
    started=time.perf_counter()
    a=load_field(args.field)
    b=load_field("mps") if args.operation=="metrics" else None
    if args.operation=="spectrum":
        if args.spectrum=="dns-uvw":
            if args.field!="dns":
                raise ValueError("three-component spectrum requires the three DNS fields")
            spectrum_fields=(a,load_field("dns_v"),load_field("dns_w"))
        else:
            zeros=np.zeros_like(a)
            spectrum_fields=(a,zeros,zeros)
    load_seconds=time.perf_counter()-started
    if args.operation=="derivative":
        old=reference(REFERENCES["derivative"]).spectral_derivative
        calculate=(lambda: old(a,args.axis,SPACING)) if args.version=="reference" else (lambda: spectral_derivative_slabbed(a,args.axis,SPACING))
    elif args.operation=="metrics":
        if args.version=="reference":
            old=reference(REFERENCES["metrics"])
            calculate=lambda: np.array([old.calculate_fidelity(a,b),old.calculate_l2norm(a,b)])
        else:
            def calculate():
                result=compare_fields_chunked(a,b)
                return np.array([result["fidelity"],result["relative_l2"]])
    else:
        if args.version=="reference":
            function=reference(REFERENCES["spectrum"],"calculate_spectrum")
            calculate=lambda: np.stack(function(*spectrum_fields))
        elif args.version=="float32":
            function=reference(REFERENCES["float32"],"calculate_spectrum")
            calculate=lambda: np.stack(function(*spectrum_fields,verbose=False))
        else:
            calculate=lambda: np.stack(calculate_spectrum_binned(*spectrum_fields))
    started=time.perf_counter()
    result=calculate()
    calculate_seconds=time.perf_counter()-started
    started=time.perf_counter()
    np.save(args.result_file,result)
    output_seconds=time.perf_counter()-started
    return {"operation":args.operation,"field":args.field,"version":args.version,
        "axis":args.axis,"load_seconds":load_seconds,"calculate_seconds":calculate_seconds,
        "output_seconds":output_seconds,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "numpy":np.__version__,"h5py":h5py.__version__,"shape":list(result.shape)}


def compare(first,second):
    import numpy as np
    a=np.load(first,mmap_mode="r")
    b=np.load(second,mmap_mode="r")
    if a.shape!=b.shape or a.dtype!=b.dtype:
        raise RuntimeError("output structure mismatch")
    absolute=relative=0.0
    bitwise=True
    zero_mismatch=nonfinite_mismatch=0
    for start in range(0,a.shape[0],8):
        x=np.asarray(a[start:start+8]).ravel()
        y=np.asarray(b[start:start+8]).ravel()
        bitwise=bitwise and bool(np.array_equal(x.view(np.uint64),y.view(np.uint64)))
        finite=np.isfinite(x)&np.isfinite(y)
        nonfinite_mismatch+=int(np.count_nonzero(~finite & ~(x==y) & ~(np.isnan(x)&np.isnan(y))))
        delta=np.abs(x[finite]-y[finite])
        if delta.size:
            absolute=max(absolute,float(delta.max()))
            nonzero=x[finite]!=0
            if nonzero.any():
                relative=max(relative,float((delta[nonzero]/np.abs(x[finite][nonzero])).max()))
        zero_mismatch+=int(np.count_nonzero((x==0)&(y!=0)))
    return {"bitwise_equal":bitwise,"max_absolute_difference":absolute,
        "max_relative_difference_nonzero_reference":relative,"zero_reference_mismatches":zero_mismatch,
        "nonfinite_mismatches":nonfinite_mismatch}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory",type=Path,required=True)
    parser.add_argument("--spectrum",choices=["dns-uvw","u-only","defer"],required=True)
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--operation",choices=["derivative","metrics","spectrum"])
    parser.add_argument("--version",choices=["reference","staged","float32"])
    parser.add_argument("--field",choices=["dns","mps"])
    parser.add_argument("--axis",type=int,default=0)
    parser.add_argument("--result-file",type=Path)
    args=parser.parse_args()
    directory=args.output_directory.resolve()
    plan={"dns":str(DNS),"reconstructed_u":str(MPS),"shape":[1024]*3,
        "dtype":"Float64","crop":"none","centering":"none at kernel boundary",
        "axes":"historical Python JHS loader transpose(2,1,0)","spacing":SPACING,
        "derivatives":"first order, all three axes, DNS and reconstructed u",
        "metrics":"squared-overlap fidelity and relative L2, DNS u versus MPS u",
        "spectrum":args.spectrum,"output_directory":str(directory),
        "timing":"one full-size call per version; load and output time reported separately",
        "temporary_outputs":"gradient arrays removed after full elementwise comparison",
        "acceptance":"differences reported; no scientific tolerance selected"}
    if args.spectrum=="dns-uvw":
        plan["dns_v"]=str(DNS_V)
        plan["dns_w"]=str(DNS_W)
        plan["spectrum_description"]="actual three-component DNS (U1,V1,W1); no MPS v/w or mixed reconstruction"
    if args.plan:
        print(json.dumps(plan,indent=2)); return
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("run in an approved compute allocation")
    if args.worker:
        result=worker(args)
        print(json.dumps(result)); return
    directory.mkdir(parents=True,exist_ok=False)
    sources={**REFERENCES,"new_reductions":ROOT/"src/python/turbulence_tools_stats/reductions.py",
             "new_spectrum":ROOT/"src/python/turbulence_tools_stats/spectrum.py"}
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}
    for name,path in sources.items():
        (directory/(name+".source.py")).write_bytes(path.read_bytes())
    report={"plan":plan,"job_id":os.environ["SLURM_JOB_ID"],"host":socket.gethostname(),
        "source_sha256":hashes,"input_stat":{str(path):{"bytes":path.stat().st_size,
            "mtime_ns":path.stat().st_mtime_ns} for path in
            ((DNS,MPS,DNS_V,DNS_W) if args.spectrum=="dns-uvw" else (DNS,MPS))},"results":[]}
    (directory/"report.json").write_text(json.dumps(report,indent=2))
    tasks=[("derivative",field,axis) for field in ("dns","mps") for axis in range(3)]
    tasks.append(("metrics","dns",0))
    if args.spectrum=="u-only":
        tasks.extend(("spectrum",field,0) for field in ("dns","mps"))
    elif args.spectrum=="dns-uvw":
        tasks.append(("spectrum","dns",0))
    for operation,field,axis in tasks:
        key=f"{operation}_{field}_{axis}"
        measurements={}
        with tempfile.TemporaryDirectory(prefix=key+"-",dir=directory) as temporary:
            for version in (["reference","staged","float32"] if operation=="spectrum" else ["reference","staged"]):
                result_path=Path(temporary)/(version+".npy")
                command=[sys.executable,"-B",str(Path(__file__).resolve()),"--output-directory",str(directory),
                    "--spectrum",args.spectrum,"--worker","--operation",operation,"--field",field,
                    "--axis",str(axis),"--version",version,"--result-file",str(result_path)]
                with (directory/(key+"_"+version+".log")).open("x") as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                measurements[version]=json.loads((directory/(key+"_"+version+".log")).read_text().splitlines()[-1])
            differences={version:compare(Path(temporary)/"reference.npy",Path(temporary)/(version+".npy"))
                for version in measurements if version!="reference"}
            if operation in ("metrics","spectrum"):
                import numpy as np
                np.savez(directory/(key+".npz"),**{version:np.load(Path(temporary)/(version+".npy")) for version in measurements})
        report["results"].append({"task":key,"measurements":measurements,"differences":differences})
        (directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(key+" compared",flush=True)
    report["sources_unchanged"]=all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name] for name,path in sources.items())
    if not report["sources_unchanged"]:
        raise RuntimeError("source changed during benchmark")
    report["state"]="COMPLETED"
    (directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
