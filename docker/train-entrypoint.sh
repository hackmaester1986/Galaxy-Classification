#!/bin/sh
set -e

if [ "$1" = "train" ]; then
  shift
fi

exec python /app/scripts/train_models.py "$@"