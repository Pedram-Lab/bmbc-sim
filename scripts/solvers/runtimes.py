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

plt.loglog(threads, implicit_direct, 'ro-')
plt.loglog(threads, [implicit_direct[0] / n for n in threads], 'b--')
plt.xlabel('Number of threads')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs. Number of Threads')
plt.grid(which='both')
plt.legend(['Runtime', 'Ideal Speedup'])
plt.show()
