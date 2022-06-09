. ./path.sh

echo "1272-135031-0009 lat1.ark:1840576" > 0009.scp

lattice-copy --write-compact=false scp:0009.scp ark:lattice.ark

lattice-copy --write-compact=true scp:0009.scp ark:compactlattice.ark