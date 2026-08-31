# Minimal NFSv4-only export server, built ON the worker at install time.
# Vendored because the published NFS server images are amd64-only while the
# kit's target hardware (DGX Spark) is aarch64 — see issue #1.
FROM alpine:3.22
RUN apk add --no-cache nfs-utils
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 2049
ENTRYPOINT ["/entrypoint.sh"]
