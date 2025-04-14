"""
Plot the runtimes of the implicit solvers over the number of threads.
"""
import matplotlib.pyplot as plt

threads = [1, 2, 4, 8, 16]

implicit_direct = [
    12.643525838851929,
    13.944251775741577,
    14.410449028015137,
    14.782570123672485,
    14.404155254364014
]

implicit_rhs = [
    11.686000108718872,
    11.859061002731323,
    12.015008926391602,
    12.254558324813843,
    12.10342025756836,
]

implicit_splitting = [
    18.74595308303833,
    19.5186710357666,
    19.945695161819458,
    20.179630041122437,
    19.774117708206177
]

cg_full = [
    4.088802099227905,
    2.052219867706299,
    1.2776226997375488,
    1.1238210201263428,
    1.6264290809631348
]

cg_smart = [
    9.626245260238647,
    4.687720775604248,
    2.8375468254089355,
    2.103302001953125,
    2.9910757541656494 
]

all_runtimes = [
    (implicit_direct, 'Implicit Direct'),
    (implicit_rhs, 'Implicit RHS'),
    (implicit_splitting, 'Implicit Splitting'),
    (cg_full, 'CG Full'),
    (cg_smart, 'CG Smart')
]

for runtimes, label in all_runtimes:
    plt.loglog(threads, runtimes, 'o-', label=label)
plt.loglog(threads, [implicit_splitting[0] / n for n in threads], 'b--', label='Ideal Speedup')
plt.xlabel('Number of threads')
plt.xticks(threads, threads)
plt.gca().xaxis.set_minor_locator(plt.NullLocator())
plt.ylabel('Runtime (s)')
plt.title('Runtime vs. Number of Threads')
plt.grid(which='both')
plt.legend()
plt.show()
