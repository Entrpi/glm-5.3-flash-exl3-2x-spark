#!/bin/sh
# Minimal NFSv4-only export server for the kit's weights share (issue #1:
# published NFS server images are amd64-only; this builds natively on the
# Spark). Env contract is compatible with the previous erichough usage:
#   NFS_EXPORT_0..NFS_EXPORT_9   exports(5) lines, e.g.
#     /export/glm53-exl3 *(ro,no_subtree_check,fsid=0,insecure)
# Serves NFSv4/4.1/4.2 only on 2049 (publish with -p; clients mount the
# fsid=0 pseudo-root as device=:/ with vers=4.x). Needs --privileged and
# the host's nfsd kernel module (/lib/modules mounted ro for modprobe).
# Field-tested pitfall (issue #1): NEVER pass rpc.nfsd --no-nfs-version 2 —
# kernels without NFSv2 hard-error on it ("2: Unsupported version"), and v2
# is absent by default on modern kernels anyway.
set -eu

modprobe nfsd 2>/dev/null || true
mountpoint -q /proc/fs/nfsd || mount -t nfsd nfsd /proc/fs/nfsd || {
  echo "FATAL: cannot mount the nfsd fs — is the host's nfsd module loadable?" >&2
  echo "       (Secure Boot hosts: run 'sudo modprobe nfsd' on the worker first)" >&2
  exit 1
}

: > /etc/exports
i=0
while [ "$i" -le 9 ]; do
  eval "line=\${NFS_EXPORT_$i:-}"
  [ -n "$line" ] && printf '%s\n' "$line" >> /etc/exports
  i=$((i + 1))
done
grep -q '[^[:space:]]' /etc/exports || { echo "FATAL: no NFS_EXPORT_<n> env set" >&2; exit 1; }

exportfs -r
rpc.mountd --no-nfs-version 3
rpc.nfsd --no-nfs-version 3 8
echo "NFSv4 export ready on :2049"
exportfs -v

stop() { rpc.nfsd 0 2>/dev/null; exportfs -ua 2>/dev/null; exit 0; }
trap stop TERM INT
while :; do sleep 3600 & wait $!; done
