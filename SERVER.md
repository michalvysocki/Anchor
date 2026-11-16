# Running experiments on the university server

The server is a shared Windows 11 machine (32 cores), reached by remote desktop. Written 2026-11-16 for moving the rest of e03's select stage there and for e03's final stage; nothing has been run on it yet.

## Why WSL2 and not Windows directly

Run names contain `*` (the M\* variants, e.g. `k0.8_gru_M*_main.json`), which Windows file systems do not allow: 40 of the select files have one. The experiments therefore run in Linux under WSL2, with the repository and its results in the Linux home directory (`~`), never under `/mnt/c`. Inside WSL2 `launch.sh` and the commands of the e03 README work unchanged.

## Facts the procedure rests on

- A job counts as done once its `.pt` is saved, and `run.py` skips it (`--force` reruns it). Copying `results/runs/<stage>/` to another machine therefore resumes the stage there; a job interrupted mid-way is rerun from the start.
- Every run JSON records the machine in its `machine` field, so which jobs ran where can be read from the results.
- The training code of e03 (`run.py`, its design module, `src/anchor`, `pyproject.toml`, `uv.lock`) has not changed since commit `6e8b589`, with which the select stage was launched on 2026-11-14; later commits change only `evaluate.py` and `gap_report.py`.
- `uv.lock` pins torch 2.8.0, with a Linux x86_64 wheel. On Linux, PyPI's torch is the CUDA build: a download of about 900 MB plus the NVIDIA libraries, unused on CPU.
- One e03 select process uses about 300 MB of memory at 2 threads.
- The Mac and the server must not run the same stage at the same time: each skips only the jobs whose `.pt` it has itself, so both would train the same jobs.

## 1. Windows, once

In PowerShell:

```
wsl --status
wsl -l -v
```

If WSL2 with a distribution is missing, `wsl --install -d Ubuntu-24.04` installs it (needs administrator rights and a restart; without them, ask the administrators of the machine).

In Windows settings:
- Power: sleep and hibernation set to never (the counterpart of `caffeinate`).
- Windows Update: updates paused for the duration of the runs, since a restart stops every process.
- Remote desktop: end a session by disconnecting (closing the window), never by signing out, which stops the processes. Windows 11 allows one interactive session at a time, so other users of the machine should know that runs are in progress.

## 2. In WSL, once

```
sudo apt update && sudo apt install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
nproc && free -g
```

Check that WSL keeps processes alive without an open terminal before relying on it: `tmux new -d -s probe 'sleep 900'`, close every WSL window and disconnect, reconnect after about 10 minutes, then `tmux ls` should still list `probe`. If it does not, leave one WSL window open and only disconnect the remote desktop.

## 3. Packing on the Mac

First stop the stage on the Mac (`Ctrl-C` in the terminal of `launch.sh`, or `kill` of its PID; the launcher stops its processes). The jobs in progress are lost and run again on the server. Then, from the repository:

```
git bundle create ~/Desktop/anchor.bundle main
tar -czf ~/Desktop/e03_select.tgz -C experiments/e03_henon_latent/results/runs select
ls experiments/e03_henon_latent/results/runs/select/*.pt | wc -l
```

The bundle carries the committed history, so the code on the server is `main` as committed, with its commit recorded; the e03 results go separately, in the tar file, since `.pt` files are not in git. The last number is the count of finished jobs, to compare on the server. On 2026-11-16 (113 jobs) the bundle had 16 MB and the tar file 122 MB, and the extracted copy kept all 40 file names with `*`.

Copy both files to the server, e.g. by sharing a Mac folder in the remote desktop client (Windows App: the connection's settings, Folders), into the Windows Downloads folder.

## 4. Unpacking in WSL

With `<user>` the Windows user name:

```
git clone /mnt/c/Users/<user>/Downloads/anchor.bundle ~/anchor
cd ~/anchor && uv sync
mkdir -p experiments/e03_henon_latent/results/runs
tar -xzf /mnt/c/Users/<user>/Downloads/e03_select.tgz -C experiments/e03_henon_latent/results/runs
ls experiments/e03_henon_latent/results/runs/select/*.pt | wc -l
git log -1 --format=%h
```

The count must equal the one on the Mac.

## 5. Checks before launching

```
uv run python experiments/e03_henon_latent/run.py select --only k1.0_L1_onestep_main
uv run python experiments/e03_henon_latent/run.py smoke --only k1.0_L1_C_main --steps 200 --threads 2
```

The first must print `done, skipped` (the copied results are seen, so nothing finished is retrained). The second trains one short run into `results/runs/smoke/` and checks that training works on this machine (about 5 s on the Mac). Both were checked on 2026-11-16 on a clone of the bundle with the extracted results, on the Mac.

## 6. Launching

```
tmux new -s e03
bash experiments/e03_henon_latent/launch.sh select <N> 2
```

Detach with `Ctrl-b d`, then disconnect the remote desktop. `<N>` processes take 2 threads each, so `<N>` = 14 uses 28 of the 32 cores; on a shared machine the number is agreed with its other users. At most 82 select jobs remain (as of 2026-11-16, fewer when the Mac is stopped later).

Progress, after `tmux attach -t e03` or from another window:

```
ls experiments/e03_henon_latent/results/runs/select/*.pt | wc -l
tail -n 3 logs/e03_select_*.log
```

The stage has 195 jobs; `launch.sh` prints `stage select done` when every process has finished.

## 7. Bringing results back to the Mac

In WSL:

```
tar -czf /mnt/c/Users/<user>/Downloads/e03_select_server.tgz -C experiments/e03_henon_latent/results/runs select
tar -czf /mnt/c/Users/<user>/Downloads/e03_select_logs_server.tgz logs
```

On the Mac, after copying both files to `~/Desktop`:

```
tar -xzf ~/Desktop/e03_select_server.tgz -C experiments/e03_henon_latent/results/runs
mkdir -p logs/server && tar -xzf ~/Desktop/e03_select_logs_server.tgz -C logs/server --strip-components 1
ls experiments/e03_henon_latent/results/runs/select/*.pt | wc -l
```

The server's copy of the results contains the Mac's runs unchanged, so extracting over them replaces them with identical files. The logs go to `logs/server/` because their names are the same as the Mac's.

## Final stage

After the select stage is complete, in the same tmux session on the server: `bash experiments/e03_henon_latent/launch.sh final <N> 2`. It reads the selected λ from the select results, which are already there. The results come back as in step 7, with `final` in place of `select`.
