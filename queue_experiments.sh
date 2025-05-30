#/bin/sh

for D in 1 2 5 10; do
 for LAMBD in 0.5 0.9 0.95 0.99; do
  echo $LAMBD $D
  ./queue_sim.py --lambd $LAMBD --d $D --n 10 --csv out.csv --max-t 100_000
 done
done
