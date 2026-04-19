"""
Part 3 - stability.py  (Mục 3.3)
Phân tích ổn định số với 2 loại ma trận:
	- Ma trận Hilbert Hn    : ill-conditioned (số điều kiện rất lớn)
	- Ma trận ngẫu nhiên SPD: well-conditioned (số điều kiện nhỏ)

"""

import sys
import os
import math
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
	sys.path.append(parent_dir)
if current_dir not in sys.path:
	sys.path.append(current_dir)

from part1.part1_skeleton import Matrix
from part3_skeleton import gaussian_solve, lu_solve, Iterative_Solver

import numpy as np
import matplotlib.pyplot as plt

DEFINE_HILBERT_SIZES	= [3, 5, 7, 9, 11, 13]
DEFINE_SPD_SIZES	= [10, 30, 50, 100]
DEFINE_GS_EPS		= 1e-10
DEFINE_GS_MAX_ITER	= 2000

def vec_sub(a, b):
	return [a[i] - b[i] for i in range(len(a))]

def norm2(v):
	return math.sqrt(sum(x*x for x in v))

def relative_error(x_approx, x_ref):
	return norm2(vec_sub(x_approx, x_ref))/norm2(x_ref)

# trả về Matrix (Part 1)
def make_hilbert(n):
	"""Ma trận Hilbert H[i][j] = 1/(i+j+1), nghiệm đúng = [1,...,1]"""
	data = [[1.0/(i + j + 1) for j in range(n)] for i in range(n)]
	a = Matrix(data)
	x_true = [1.0] * n
	b = [sum(a[i][j] for j in range(n)) for i in range(n)]
	return a, b, x_true


def make_random_spd(n, seed=42):
	"""Ma trận SPD ngẫu nhiên: A = B^T*B + n*I"""
	rng = random.Random(seed)
	b_mat = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
	data = [[sum(b_mat[k][i]*b_mat[k][j] for k in range(n)) for j in range(n)]
	        for i in range(n)]
	for i in range(n):
		data[i][i] += n
	b = [rng.gauss(0, 1) for _ in range(n)]
	return Matrix(data), b


def make_diag_dominant(n, seed=0):
	"""Ma trận chéo trội chặt — dùng cho Gauss-Seidel"""
	rng = random.Random(seed)
	data = [[0.0]*n for _ in range(n)]
	for i in range(n):
		off_sum = 0.0
		for j in range(n):
			if i != j:
				data[i][j] = rng.uniform(-1, 1)
				off_sum += abs(data[i][j])
		data[i][i] = off_sum + 1.0
	b = [rng.uniform(-10, 10) for _ in range(n)]
	return Matrix(data), b


# 3.3 Stability analysis

def stability_analysis():
	hilbert_data	= {"sizes": [], "conds": [], "err_gauss": [], "err_lu": []}
	spd_data	= {"sizes": [], "conds": [], "err_gauss": [], "err_lu": [], "err_gs": []}

	# ── [A] Hilbert ──────────────────────────
	print("[A] Ma trận Hilbert (ill-conditioned)")
	print(f"{'n':>4}  {'Cond(H)':>14}  {'Err Gauss':>12}  {'Err LU':>12}")
	print("-" * 48)

	for n in DEFINE_HILBERT_SIZES:
		a, b, x_true = make_hilbert(n)
		cond = np.linalg.cond(np.array(a.data))

		try:
			err_gauss = relative_error(gaussian_solve(a, b), x_true)
		except Exception:
			err_gauss = float('inf')

		try:
			err_lu = relative_error(lu_solve(a, b), x_true)
		except Exception:
			err_lu = float('inf')

		hilbert_data["sizes"].append(n)
		hilbert_data["conds"].append(cond)
		hilbert_data["err_gauss"].append(err_gauss)
		hilbert_data["err_lu"].append(err_lu)
		print(f"{n:>4}  {cond:>14.3e}  {err_gauss:>12.3e}  {err_lu:>12.3e}")

	# ── [B] SPD ──────────────────────────────
	print("\n[B] Ma trận SPD ngẫu nhiên (well-conditioned)")
	print(f"{'n':>4}  {'Cond(A)':>12}  {'Err Gauss':>12}  {'Err LU':>12}  {'Err GS':>12}")
	print("-" * 58)

	for n in DEFINE_SPD_SIZES:
		a_spd, b_spd	= make_random_spd(n)
		a_dd,  b_dd	= make_diag_dominant(n)

		cond		= np.linalg.cond(np.array(a_spd.data))
		x_ref		= np.linalg.solve(np.array(a_spd.data), np.array(b_spd)).tolist()
		x_ref_dd	= np.linalg.solve(np.array(a_dd.data),  np.array(b_dd)).tolist()

		err_gauss	= relative_error(gaussian_solve(a_spd, b_spd), x_ref)
		err_lu		= relative_error(lu_solve(a_spd, b_spd),      x_ref)

		solver		= Iterative_Solver(a_dd, b_dd)
		x_gs, _		= solver.gauss_seidel(eps=DEFINE_GS_EPS, max_iterations=DEFINE_GS_MAX_ITER)
		err_gs		= relative_error(x_gs, x_ref_dd)

		spd_data["sizes"].append(n)
		spd_data["conds"].append(cond)
		spd_data["err_gauss"].append(err_gauss)
		spd_data["err_lu"].append(err_lu)
		spd_data["err_gs"].append(err_gs)
		print(f"{n:>4}  {cond:>12.3e}  {err_gauss:>12.3e}  {err_lu:>12.3e}  {err_gs:>12.3e}")

	return hilbert_data, spd_data


# Plotting
def plot_stability(hilbert_data, spd_data):
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))

	ax = axes[0]
	ax.plot(hilbert_data["sizes"], hilbert_data["err_gauss"], marker='s', label="Gauss", color="steelblue",  linewidth=2)
	ax.plot(hilbert_data["sizes"], hilbert_data["err_lu"],    marker='^', label="LU",    color="darkorange", linewidth=2)
	ax.set_yscale("log")
	ax.set_xlabel("n")
	ax.set_ylabel("Sai số tương đối")
	ax.set_title("Ma trận Hilbert (ill-conditioned)")
	ax.legend()
	ax.grid(True, alpha=0.3)

	ax = axes[1]
	ax.plot(spd_data["sizes"], spd_data["err_gauss"], marker='s', label="Gauss",        color="steelblue",  linewidth=2)
	ax.plot(spd_data["sizes"], spd_data["err_lu"],    marker='^', label="LU",           color="darkorange", linewidth=2)
	ax.plot(spd_data["sizes"], spd_data["err_gs"],    marker='o', label="Gauss-Seidel", color="seagreen",   linewidth=2)
	ax.set_yscale("log")
	ax.set_xlabel("n")
	ax.set_ylabel("Sai số tương đối")
	ax.set_title("Ma trận SPD ngẫu nhiên (well-conditioned)")
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.suptitle("Phân tích ổn định số", fontsize=13)
	plt.tight_layout()
	plt.savefig("stability_analysis.png", dpi=150, bbox_inches="tight")
	plt.close()
	print("\n[OK] stability_analysis.png")

# Main
if __name__ == "__main__":
	hilbert_data, spd_data = stability_analysis()
	plot_stability(hilbert_data, spd_data)