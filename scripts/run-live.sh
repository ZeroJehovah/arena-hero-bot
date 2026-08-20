#!/usr/bin/env bash

set -uo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p var

if (($# == 0)); then
    set -- --target-workers 12 --no-max-population --log-level INFO
fi

while true; do
    env \
        -u all_proxy \
        -u http_proxy \
        -u https_proxy \
        -u ALL_PROXY \
        -u HTTP_PROXY \
        -u HTTPS_PROXY \
        uv run arena-hero-bot "$@" 2>&1 | tee -a var/bot.log
    bot_status=${PIPESTATUS[0]}
    printf '%s WARNING live bot exited with status %d; restarting in 5 seconds\n' \
        "$(date -Is)" "$bot_status" | tee -a var/bot.log
    sleep 5
done
