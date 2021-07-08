#!/bin/bash

curl http://localhost:1234/invocations \
  -H 'Content-Type: application/json; format=pandas-records' \
  -d @$1
