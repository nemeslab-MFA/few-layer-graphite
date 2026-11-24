"""5 x 5 grid of CPot x n_sub, full bias sweeps for N = 3 and 4 (4 processes)."""
import multiprocessing as mp, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgfit as R
CP = (-0.3, -0.15, 0.0, 0.15, 0.3)
NS = (-0.5e13, 0.0, 0.5e13)
if __name__ == "__main__":
    R.threads1()
    os.makedirs(R.OUT, exist_ok=True)
    jobs = [(N, c, n) for N in (3, 4) for c in CP for n in NS]
    jobs.sort(key=lambda j: (abs(j[1]) + abs(j[2]) / 1e13, j[0]))      # centre first
    with mp.get_context("spawn").Pool(4) as pool:
        list(pool.imap_unordered(R.sweep, jobs))
