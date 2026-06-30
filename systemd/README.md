# memo systemd units (Linux)

Linux counterparts of memo's macOS launchd agents. memo's `install-watcher`
and the nightly `dream` LaunchAgent are macOS-only; on Linux use these **user**
units instead (no root required).

| Unit | Purpose | macOS equivalent |
|---|---|---|
| `memo-dream.service` + `memo-dream.timer` | nightly self-maintenance at 03:00 | `launchd/com.memo.dream.plist` |
| `memo-watch.service` | auto-reindex on file change | `memo install-watcher` |

## Install

```bash
mkdir -p ~/.config/systemd/user
cp systemd/memo-dream.service systemd/memo-dream.timer systemd/memo-watch.service \
   ~/.config/systemd/user/
systemctl --user daemon-reload

# nightly maintenance (enable the TIMER, not the service):
systemctl --user enable --now memo-dream.timer

# file-watcher (optional):
systemctl --user enable --now memo-watch.service
```

The units call `%h/.local/bin/memo` (where `pipx`/`uv tool` install it). If your
`memo` lives elsewhere (`command -v memo`), edit `ExecStart` accordingly.

To run maintenance/watch when you're logged out, enable lingering once:

```bash
sudo loginctl enable-linger "$USER"
```

## Inspect

```bash
systemctl --user list-timers memo-dream.timer
systemctl --user start memo-dream.service     # run a maintenance pass now
journalctl --user -u memo-dream.service -n 50  # logs
```

> First run note: the CPU backend downloads the embedding model on first use, so
> run `memo search <anything>` once **online** before relying on the offline
> nightly timer.
