#!/bin/bash
source ../isaaclabenv/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Ithor-v0 --num_envs 128 --max_iterations 2000 --headless
