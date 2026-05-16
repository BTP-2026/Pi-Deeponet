"""
remote_run.py — transfers files and launches training on GPU server via SSH.
"""
import os
import paramiko
from pathlib import Path

HOST     = "10.13.3.3"
USER     = "amrit"
PASSWORD = "amrit@1234"
REMOTE   = "/home/amrit/btp"

FILES = [
    "pideeponet_2d_dirichlet_v0_cuda.py",
    "networks.py",
    "Surface_Solution.txt",
    "data_v0.txt",
]

CMD = (
    f"mkdir -p {REMOTE}/output_v0_run5 && "
    f"cd {REMOTE} && "
    f"nohup python3 -u pideeponet_2d_dirichlet_v0_cuda.py "
    f"  --epochs 10000 --p_dim 128 --branch_h 64 64 --trunk_h 128 128 "
    f"  --n_fourier 8 --output_scale 20.0 "
    f"  --w_d 500 --w_neu 10 --warmup_epochs 500 "
    f"  --train_step 1.0 --device cuda "
    f"  --lr 1e-3 --log_dir output_v0_run5 "
    f"  --no_progress "
    f"> output_v0_run5/run.log 2>&1 & echo PID:$!"
)

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST} ...")
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.")

    # ---- check GPU ----
    _, out, _ = client.exec_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1")
    gpu_info = out.read().decode().strip()
    print(f"GPU: {gpu_info}")

    # ---- check python / torch ----
    _, out, err = client.exec_command("python3 -c \"import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())\" 2>&1")
    torch_info = out.read().decode().strip() + err.read().decode().strip()
    print(f"PyTorch: {torch_info}")

    # ---- transfer files ----
    sftp = client.open_sftp()
    try:
        sftp.stat(REMOTE)
    except FileNotFoundError:
        sftp.mkdir(REMOTE)

    local_dir = Path(__file__).parent
    for fname in FILES:
        local_path = local_dir / fname
        remote_path = f"{REMOTE}/{fname}"
        print(f"  Uploading {fname} ({local_path.stat().st_size / 1024:.0f} KB) ...", end=" ")
        sftp.put(str(local_path), remote_path)
        print("done")
    sftp.close()

    # ---- launch training ----
    print(f"\nLaunching training ...")
    _, out, err = client.exec_command(CMD, get_pty=False)
    stdout = out.read().decode().strip()
    stderr = err.read().decode().strip()
    print(f"stdout: {stdout}")
    if stderr:
        print(f"stderr: {stderr}")

    client.close()
    print("\nDone. Training is running in background on the GPU server.")
    print(f"Check progress: ssh {USER}@{HOST} 'tail -f {REMOTE}/output_v0_run5/run.log'")

if __name__ == "__main__":
    main()
