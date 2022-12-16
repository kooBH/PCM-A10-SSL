#!/bin/bash

DEVICE=cuda:0

list="$(ps -aux)"

#echo "$list"

while read -r p; do
  if grep -q "${DEVICE}" <<< $p; then
    a=${p[0]}
    idx=0
    for i in $a; do
      if [ $idx -eq 1 ]
        then
#          echo $i
           sudo kill -9 $i
           echo [kill] ${p}
      fi
      idx=$(($idx+1))
    done
  fi
done <<<"$list"

#echo =======
