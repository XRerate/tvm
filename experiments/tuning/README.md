# how to build?
Basically you need to build two things: one for the remote server (your server machine) and one for the client (android device).
TBD

# how to run?
I assume there are three machines: local (your desktop), server (remote server), and client (android device).
I'll refer to the remote server as `$remote_server`.

First, setup ports.
Run the following commands on your local machine:
```bash
# SSH tunnels
ssh -N -R 5037:127.0.0.1:5037 $remote_server    # forward ADB port
ssh -N -L 9190:127.0.0.1:9190 $remote_server    # server -> client port
ssh -N -R 9090:127.0.0.1:9090 $remote_server    # client -> server port (for tuning)
ssh -N -R 9091:127.0.0.1:9091 $remote_server    # client -> server port (for benchmarking). 

# ADB port forwarding/reversing
adb reverse tcp:9190 tcp:9190                   # client -> server port
adb forward tcp:9090 tcp:9090                   # server -> client port (for tuning)
adb forward tcp:9091 tcp:9091                   # server -> client port (for benchmarking)
adb forward --list                               # check the port forwarding
adb reverse --list                              # check the port reversing

# Maybe it isn't necessary to separate sessions between tuning and benchmarking!
```

Next, run tracker server on the remote server:
```bash
conda activate tvm-build-venv
python -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190
```

Then, run RPC server on the android device:
```bash
# Use two different terminal sessions for each command
adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/tvm_rpc server --port=9090 --tracker=127.0.0.1:9190 --key=android64"
adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/tvm_rpc server --port=9091 --tracker=127.0.0.1:9190 --key=android64_bench"
```

Now you can run tuning or benchmarking scripts on your remote server.
For tuning:
```bash
python3 rpc_matmul_autotune.py --config configs/cpu.toml # android cpu tuning
python3 rpc_matmul_autotune.py --config configs/opencl.toml # android gpu opencl backend tuning (vulkan is not tested yet)
```

For benchmarking:
```bash
python3 run_so.py --backend cpu --adb-label --so tuning_logs/cpu_matvec/1x1536x2048_cand001_base.so --number 150 --repeat 100
python3 run_so.py --backend cpu --adb-label --dir tuning_logs/cpu_large_gemv/ --csv cpu_gemv_large.csv
```

For more details, run with `--help` flag.