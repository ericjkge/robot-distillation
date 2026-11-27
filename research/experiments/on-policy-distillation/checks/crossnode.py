"""Two Slurm jobs, one Ray cluster: the check that OPD's roles can run as separate single-node jobs on O2.
    python crossnode.py head <address file> | python crossnode.py join <address file>
The head job starts a Ray head and writes its address; the other job joins it. The head then runs one GPU actor on
each node and an NCCL all-reduce between them (as the trainer's weight sync to the rollout replicas uses NCCL across
machines), and prints both nodes and the result."""

import os
import socket
import subprocess
import sys
import time

import ray

role, address_file = sys.argv[1], sys.argv[2]
ray_bin = os.path.join(os.path.dirname(sys.executable), "ray")
ip = socket.gethostbyname(socket.gethostname())
if role == "join":
    while not os.path.exists(address_file):
        time.sleep(5)
    address = open(address_file).read().strip()
    print(f"{socket.gethostname()} joins {address}", flush=True)
    subprocess.run([ray_bin, "start", f"--address={address}", "--num-gpus=1", "--block"], check=False)
    sys.exit(0)

port = 6390
subprocess.run([ray_bin, "start", "--head", f"--port={port}", "--num-gpus=1", "--min-worker-port=0", "--max-worker-port=0"], check=True)
open(address_file, "w").write(f"{ip}:{port}")
ray.init(address=f"{ip}:{port}")
deadline = time.time() + 3600
while sum(n["Alive"] for n in ray.nodes()) < 2:
    assert time.time() < deadline, "the second job never joined"
    time.sleep(5)
print("nodes:", [n["NodeManagerHostname"] for n in ray.nodes() if n["Alive"]], flush=True)


@ray.remote(num_gpus=1)
class Rank:
    def host(self):
        return socket.gethostname(), socket.gethostbyname(socket.gethostname())

    def allreduce(self, rank, master):
        import torch
        import torch.distributed as dist

        dist.init_process_group("nccl", init_method=f"tcp://{master}:29511", rank=rank, world_size=2)
        x = torch.full((256 * 1024 * 1024,), float(rank + 1), device="cuda")  # 1 GiB, as a weight sync's buckets
        torch.cuda.synchronize()
        start = time.time()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        return x[0].item(), round(1.0 / (time.time() - start), 2)


ranks = [Rank.options(scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote()
         for n in ray.nodes() if n["Alive"]]
hosts = ray.get([r.host.remote() for r in ranks])
print("actors on:", hosts, flush=True)
print("all_reduce (sum 3.0 expected, GiB/s):", ray.get([r.allreduce.remote(i, hosts[0][1]) for i, r in enumerate(ranks)]), flush=True)
