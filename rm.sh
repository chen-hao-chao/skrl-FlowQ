for d in "runs/results_sac_*"; do
  for s in $d; do
    for z in $s; do
      find $z -type f -name 'agent_*' -exec rm {} \;
    done
  done
done