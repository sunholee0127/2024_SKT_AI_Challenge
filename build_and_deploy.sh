#!/bin/bash

docker buildx build --no-cache --push --platform linux/amd64 -f ./Dockerfile -t 337390186135.dkr.ecr.ap-northeast-2.amazonaws.com/stg-ecr-golf-aihl:tbrandchatbotv0.0.1 .\
&& kubectl apply -f ./deployment.yaml