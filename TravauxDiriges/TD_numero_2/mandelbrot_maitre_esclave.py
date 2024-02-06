import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log, ceil
from time import time
from mpi4py import MPI
import matplotlib.cm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        z: complex
        iter: int

        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations

        z = 0
        for iter in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z))) / log(2)
                return iter
        return self.max_iterations

def calculate_partial_mandelbrot(start_row, end_row, width, height, mandelbrot_set, scaleX, scaleY):
    partial_convergence = np.empty((width, end_row - start_row), dtype=np.double)

    for y in range(start_row, end_row):
        for x in range(width):
            c = complex(-2. + scaleX * x, -1.125 + scaleY * y)
            partial_convergence[x, y - start_row] = mandelbrot_set.convergence(c, smooth=True)

    return partial_convergence

def master_task(width, height, mandelbrot_set, scaleX, scaleY):
    rows_per_process = ceil(height / (size - 1))  
    all_results = []

    deb = time()

    # Distribute rows to slave processes
    for i in range(1, size):
        start_row = (i - 1) * rows_per_process
        end_row = min(i * rows_per_process, height)
        comm.send((start_row, end_row), dest=i)

    # Gather results from slave processes
    for i in range(1, size):
        partial_result = comm.recv(source=i)
        all_results.append(partial_result)

    fin = time()
    print(f"Master: Time taken for computation: {fin - deb}")

    # Process master gathers results and saves the image
    convergence = np.concatenate(all_results, axis=1)
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))
    fin = time()
    print(f"Master: Time taken for image creation: {fin - deb}")

    # Save the image
    image.save('mandelbrot_image_mpi_maitre_esclave.png')

def slave_task(mandelbrot_set, scaleX, scaleY):
    # Receive rows to calculate from master
    start_row, end_row = comm.recv(source=0)

    # Calculate partial Mandelbrot for assigned rows
    deb = time()
    partial_result = calculate_partial_mandelbrot(start_row, end_row, width, height, mandelbrot_set, scaleX, scaleY)
    fin = time()
    print(f"Slave {rank}: Time taken for computation: {fin - deb}")

    # Send partial result back to master
    comm.send(partial_result, dest=0)

if __name__ == "__main__":
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024

    scaleX = 3. / width
    scaleY = 2.25 / height

    if rank == 0:
        master_task(width, height, mandelbrot_set, scaleX, scaleY)
    else:
        slave_task(mandelbrot_set, scaleX, scaleY)
