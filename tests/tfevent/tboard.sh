#!/bin/bash
docker run -d --restart always \
    -v /u01/runs:/app/runs/:ro \
    -p 6006:6006 \
    -w "/app/" --name "tensorboard" \
    schafo/tensorboard 
