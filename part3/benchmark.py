import time
import numpy as np
import matplotlib.pyplot as plt

# Import các module từ project của bạn
from part1.gauss import gaussian_eliminate
from part2.decomposition import svd
from part2.diagonalization import diagonalize
from part3.solvers import Iterative_Solver
from part1.part1_skeleton import Matrix

# --- CẤU HÌNH HỆ THỐNG ---
METHODS = ["Gauss", "SVD", "Gauss-Seidel"]
TEST_SIZES = [50, 100, 150, 200, 500, 1000]
TRIALS = 5  # Số lần chạy để lấy trung bình
EPS = 1e-12

def solve_svd(A_input, b_input):
    """
    Giải Ax = b bằng SVD (A = U * Sigma * V^T).
    Sử dụng vòng lặp Python thuần để xử lý dữ liệu từ đối tượng Matrix.
    """
    # Bước 1: Chuẩn hóa đầu vào
    if isinstance(A_input, np.ndarray):
        A_obj = Matrix(A_input.tolist())
    else:
        A_obj = A_input
        
    b_vec = b_input.tolist() if isinstance(b_input, np.ndarray) else b_input
    
    # Bước 2: Gọi hàm phân rã SVD tự cài đặt
    u_obj, sigma_obj, vt_obj = A_obj.svd()
    
    U_data = u_obj.data
    S_data = sigma_obj.data
    VT_data = vt_obj.data
    
    n_rows = len(U_data)
    n_cols = len(U_data[0]) if n_rows > 0 else 0
    
    # Bước 3: Tính c = U^T * b 
    # (Nhân ma trận chuyển vị của U với vector b)
    c = []
    for j in range(n_cols):
        sum_val = 0.0
        for i in range(n_rows):
            sum_val += U_data[i][j] * b_vec[i]
        c.append(sum_val)
        
    # Bước 4: Giải Sigma * y = c
    # (Sigma là ma trận đường chéo)
    y = []
    for i in range(len(c)):
        diag_val = S_data[i][i]
        if abs(diag_val) > EPS:
            y.append(c[i] / diag_val)
        else:
            y.append(0.0)
            
    # Bước 5: Tính x = V * y 
    # (V là chuyển vị của VT, nên x_j = sum(VT_ij * y_i))
    vt_rows = len(VT_data)
    vt_cols = len(VT_data[0]) if vt_rows > 0 else 0
    
    x_final = []
    for j in range(vt_cols):
        sum_val = 0.0
        for i in range(vt_rows):
            sum_val += VT_data[i][j] * y[i]
        x_final.append(sum_val)
        
    return np.array(x_final)

def generate_dominant_matrix(n):
    """Tạo ma trận chéo trội để đảm bảo hội tụ."""
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0
    return A

def calculate_error(A, x, b):
    """Tính sai số tương đối ||Ax - b|| / ||b||."""
    if x is None or np.isnan(x).any():
        return 1.0
    res = np.dot(A, x) - b
    return np.linalg.norm(res) / (np.linalg.norm(b) + EPS)

def run_benchmark():
    """Chạy benchmark và vẽ đồ thị."""
    results = {m: {"times": [], "errors": []} for m in METHODS}
    

    print(f"{'n':>5} | {'Phương pháp':<15} | {'Thời gian (s)':<12} | {'Sai số':<10}")
    print("-" * 55)

    for n in TEST_SIZES:
        A_np = generate_dominant_matrix(n)
        b_np = np.random.rand(n)
        

        A_list = A_np.tolist()
        b_list = b_np.tolist()

        for method in METHODS:
            trial_times = []
            x_sol = None

            for _ in range(TRIALS):
                start = time.perf_counter()
                try:
                    if method == "Gauss":
                        _, res_list, _ = gaussian_eliminate(A_list, b_list)
                        x_sol = np.array(res_list)
                    elif method == "SVD":
                        x_sol = solve_svd(A_np, b_np)
                    elif method == "Gauss-Seidel":
                        solver = Iterative_Solver(Matrix(A_list), b_list)
                        res_list, _ = solver.gauss_seidel()
                        x_sol = np.array(res_list)
                except Exception as e:
                    print(f"\nLỗi {method} n={n}: {e}")
                    x_sol = None
                
                trial_times.append(time.perf_counter() - start)
            
            avg_time = sum(trial_times) / len(trial_times)
            err = calculate_error(A_np, x_sol, b_np)
            
            results[method]["times"].append(avg_time)
            results[method]["errors"].append(err)
            
            print(f"{n:5d} | {method:<15} | {avg_time:12.4f} | {err:.2e}")

    # --- Vẽ biểu đồ ---
    plt.figure(figsize=(10, 6))
    for m in METHODS:
        plt.loglog(TEST_SIZES, results[m]["times"], marker='o', label=m)
    
    t0 = results["Gauss"]["times"][0]
    n0 = TEST_SIZES[0]
    theoretical = [t0 * (n / n0)**3 for n in TEST_SIZES]
    plt.loglog(TEST_SIZES, theoretical, '--', color='gray', alpha=0.5, label="Lý thuyết O(n³)")

    plt.xlabel("Kích thước ma trận (n)")
    plt.ylabel("Thời gian chạy (giây)")
    plt.title("So sánh hiệu năng các phương pháp giải hệ phương trình")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_benchmark()