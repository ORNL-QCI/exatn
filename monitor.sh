#!/bin/bash
#Will monitor the total Flop count and current memory usage:
watch --interval 1 "tail --lines=1024 ./exatn_exec_thread.0.log | grep usage | tail --lines=1"
