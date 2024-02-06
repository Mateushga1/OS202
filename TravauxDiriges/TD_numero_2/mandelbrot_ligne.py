# Calcul de l'ensemble de Mandelbrot en python
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

if __name__ == "__main__":
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024

    scaleX = 3. / width
    scaleY = 2.25 / height

    rows_per_process = ceil(height / size)
    start_row = rank * rows_per_process
    end_row = min((rank + 1) * rows_per_process, height)

    deb = time()
    partial_result = calculate_partial_mandelbrot(start_row, end_row, width, height, mandelbrot_set, scaleX, scaleY)
    fin = time()
    print(f"Process {rank}: Time taken for calculation: {fin - deb}")

    # Synchronize all processes before proceeding to the next step
    comm.Barrier()

    # Gather all partial results to process 0
    all_results = comm.gather(partial_result, root=0)

    # Process 0 concatenates results and saves the image
    if rank == 0:
        # Calculate the total time taken for computation
        total_calculation_time = time() - deb
        print(f"Total calculation time: {total_calculation_time}")

        convergence = np.concatenate(all_results, axis=1)
        
        # Calculate the time for image creation only once
        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))
        fin = time()
        print(f"Process 0: Time taken for image creation: {fin - deb}")

        # Save the image
        image.save('mandelbrot_image_mpi.png')