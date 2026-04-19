"""
manim_show.py
Visualization of SVD and Diagonalization
Run with:
    manim -pqh manim_show.py demo_video
"""

from manim import *
import numpy as np


# ---------------------------------------------
# Matrix A used throughout the video
# ---------------------------------------------
A_MAT = np.array([[3, 1],
                  [1, 2]], dtype=float)

# -- SVD --------------------------------------
U_mat, S_vals, Vt_mat = np.linalg.svd(A_MAT)
Sigma_mat = np.diag(S_vals)

# -- Eigendecomposition ------------------------
eigvals, eigvecs = np.linalg.eig(A_MAT)
# Sort by descending eigenvalue
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
P_mat   = eigvecs
P_inv   = np.linalg.inv(P_mat)
D_mat   = np.diag(eigvals)


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def mixed_tex(s, font_size=28, color=WHITE, math_color=None, line_spacing=0):
    """
    Parse a string with $$...$$ blocks:
    - normal text -> Tex
    - math inside $$ -> MathTex
    Returns a VGroup arranged horizontally
    """

    parts = [p for p in s.split("$$") if p.strip()]
    mobjects = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Normal text
            if part.strip():
                mobjects.append(
                    Text(part, font_size=font_size, color=color, line_spacing=line_spacing)
                )
        else:
            # Math part
            math_obj = MathTex(part, font_size=font_size + 3)
            if math_color:
                math_obj.set_color(math_color)
            else:
                math_obj.set_color(color)
            mobjects.append(math_obj)

    return VGroup(*mobjects).arrange(RIGHT, buff=0.15)


def fmt(x, decimals=2):
    """Format a float nicely."""
    return f"{x:.{decimals}f}"


def np_to_manim_matrix(mat, decimals=2, h_buff=1.7, v_buff=0.7, **kwargs):
    """Convert a 2-D numpy array to a Manim Matrix mobject."""
    entries = [[fmt(mat[r, c], decimals) for c in range(mat.shape[1])]
               for r in range(mat.shape[0])]
    return Matrix(entries, h_buff=h_buff, v_buff=v_buff, **kwargs)


def small_matrix(mat, decimals=2, scale=0.55, h_buff=1.7, v_buff=0.7, color=WHITE):
    m = np_to_manim_matrix(
        mat,
        decimals=decimals,
        element_to_mobject_config={"color": color},
        h_buff=h_buff,
        v_buff=v_buff,
        )
    m.scale(scale)
    return m


# ---------------------------------------------
# Scene 1 - Title
# ---------------------------------------------
class S01_Title(Scene):
    def construct(self):
        title = Text("Visualization of SVD\nand Diagonalization",
                     font_size=52, color=YELLOW,
                     line_spacing=1.3).move_to(ORIGIN)
        self.play(Write(title), run_time=2)
        self.wait(0.5)
        self.play(title.animate.to_edge(UP).scale(0.6))
        members_name = Text(
                       "Khúc Minh Quân         \n"
                       "Phạm Nguyễn Quang Sáng \n"
                       "Nguyễn Lê Anh Kiên     \n"
                       "Lê Công Minh Nhựt      \n"
                       "Nguyễn Đức Quân        ", font_size=20)
        members_mssv = Text(
                        "- 24120126\n"
                        "- 24120132\n"
                        "- 24120196\n"
                        "- 24120211\n"
                        "- 24120218"
                        ,font_size=20)
        grp_members = VGroup(members_name, members_mssv).arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=1)
        self.play(FadeIn(members))
        self.wait(2)
        self.play(FadeOut(members))

        # -- Quick teaser: Rotate -> Scale -> Rotate on unit disk --
        ax = Axes(x_range=[-2.2, 2.2, 1], y_range=[-2.2, 2.2, 1],
                  x_length=4, y_length=4,
                  axis_config={"include_tip": False}).scale(0.7).move_to(ORIGIN + DOWN*0.5)

        self.play(Create(ax))

        circle = Circle(radius=1, color=GRAY, fill_opacity=0.35).move_to(ax.c2p(0, 0))
        circle.scale(ax.get_x_unit_size())

        v1 = Arrow(ax.c2p(0,0), ax.c2p(1,0), buff=0, color=RED)
        v2 = Arrow(ax.c2p(0,0), ax.c2p(0,1), buff=0, color=BLUE)

        svd_label = mixed_tex("SVD: Rotate $$\\rightarrow$$ Scale $$\\rightarrow$$ Rotate", font_size=25, color=WHITE)\
                        .next_to(title, DOWN, buff=0.2)

        self.play(FadeIn(circle), GrowArrow(v1), GrowArrow(v2), Write(svd_label))
        self.wait(0.5)

        base_pts = [
            np.array([np.cos(t), np.sin(t)])
            for t in np.linspace(0, 2 * PI, 100)
        
        ]
        
        def make_transformed_disk(M):
            pts = [ax.c2p(*(M @ p)) for p in base_pts]
            disk = Polygon(*pts, color=GRAY, fill_opacity=0.35)
            return disk
        
        def make_basis_arrows(M):
            e1 = M @ np.array([1.0, 0.0])
            e2 = M @ np.array([0.0, 1.0])
            a1 = Arrow(ax.c2p(0, 0), ax.c2p(*e1), buff=0, color=RED)
            a2 = Arrow(ax.c2p(0, 0), ax.c2p(*e2), buff=0, color=BLUE)
            return a1, a2

        # Step 1
        step_label = mixed_tex("Step 1: $$V^T$$  (rotation)", font_size=23, color=YELLOW)\
                            .to_edge(DOWN, buff=0)
        disk2 = make_transformed_disk(Vt_mat)
        nv1, nv2 = make_basis_arrows(Vt_mat)
        self.play(
            Write(step_label),
            Transform(circle, disk2),
            Transform(v1, nv1),
            Transform(v2, nv2),
            run_time=1.5
        )
        self.wait(0.4)
        
        # Step 2
        step_label2 = mixed_tex("Step 2: $$\Sigma$$  (scale)", font_size=23, color=YELLOW)\
                            .to_edge(DOWN, buff=0)
        disk3 = make_transformed_disk(Sigma_mat @ Vt_mat)
        nv1b, nv2b = make_basis_arrows(Sigma_mat @ Vt_mat)
        self.play(
            Transform(step_label, step_label2),
            Transform(circle, disk3),
            Transform(v1, nv1b),
            Transform(v2, nv2b),
            run_time=1.5
        )
        self.wait(0.4)
        
        # Step 3
        step_label3 = mixed_tex("Step 3: $$U$$  (rotation)", font_size=23, color=YELLOW)\
                          .to_edge(DOWN, buff=0)
        disk4 = make_transformed_disk(U_mat @ Sigma_mat @ Vt_mat)
        nv1c, nv2c = make_basis_arrows(U_mat @ Sigma_mat @ Vt_mat)
        self.play(
            Transform(step_label, step_label3),
            Transform(circle, disk4),
            Transform(v1, nv1c),
            Transform(v2, nv2c),
            run_time=1.5
        )
        self.wait(0.6)

        diag_note = mixed_tex("Diagonalization:", font_size=25, color=GREEN_B)\
                        .next_to(title, DOWN, buff=0.2)
        self.play(FadeOut(i) for i in [step_label, v1, v2, ax, circle])
        self.play(Transform(svd_label, diag_note))
        diag_intro = MathTex("A = P\\cdot D \\cdot P^{-1}", font_size=46, color=BLUE)\
                            .move_to(ORIGIN)
        self.play(FadeIn(diag_intro))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 2 - Introduce SVD section
# ---------------------------------------------
class S02_SVD_Intro(Scene):
    def construct(self):
        heading = Text("Singular Value Decomposition  (SVD)", font_size=46, color=YELLOW)\
                        .to_edge(UP)
        self.play(Write(heading))

        line1 = Text("We will analyze matrix A into the product of 3 matrices:", font_size=30)
        line2 = MathTex(
            "A  =  U \\cdot \\Sigma \\cdot V^T",
            font_size=30,
            color=BLUE
        )
        intro = VGroup(line1, line2)\
            .arrange(DOWN, aligned_edge=UP, buff=0.3)\
            .next_to(heading, DOWN, buff=0.5)

        self.play(FadeIn(intro, shift=DOWN*0.3))

#        A_label = MathTex(r"A = \begin{bmatrix}3&1\\1&2\end{bmatrix}",
#                            font_size=44, color=WHITE)\
#                            .next_to(intro, DOWN, buff=0.6)
        A_label = VGroup(Text("Let", color=PINK, font_size=28), MathTex("A =", color=BLUE), small_matrix(scale=1.0, mat=A_MAT, color=WHITE))\
                        .arrange(RIGHT, buff=0.2)

        self.play(Write(A_label))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 3 - Show full SVD decomposition
# ---------------------------------------------
class S03_SVD_Decompose(Scene):
    def construct(self):
        heading = Text("SVD of matrix A", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Write(heading))

        # A = U Σ V^T  symbolic
        eq = MathTex(r"A", r"=", r"U", r"\Sigma", r"V^T",
                        font_size=48)
        eq.set_color_by_tex("U", BLUE)
        eq.set_color_by_tex(r"\Sigma", GREEN)
        eq.set_color_by_tex(r"V^T", RED)
        eq.next_to(heading, DOWN, buff=0.5)
        self.play(Write(eq))
        self.wait(0.8)

        # Expand matrices
        mA  = np_to_manim_matrix(A_MAT)
        mU  = np_to_manim_matrix(U_mat, decimals=3)
        mSig= np_to_manim_matrix(Sigma_mat, decimals=3)
        mVt = np_to_manim_matrix(Vt_mat, decimals=3)

        mU.set_color(BLUE)
        mSig.set_color(GREEN)
        mVt.set_color(RED)

        # Labels
        lA  = MathTex("A=",  font_size=36)
        lU  = MathTex("U=",  font_size=36, color=BLUE)
        lS  = MathTex(r"\Sigma=", font_size=36, color=GREEN)
        lVt = MathTex("V^T=", font_size=36, color=RED)

        row = VGroup(mA, MathTex("="), mU, mSig, mVt)\
                .arrange(RIGHT, buff=0.22).scale(0.62).next_to(eq, DOWN, buff=0.55)
        self.play(FadeIn(row, shift=UP*0.2))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 4 - Unit disk transformation step by step
# ---------------------------------------------
class S04_SVD_Transform(Scene):
    def construct(self):
        heading = Text("SVD: transformation on unit disk", font_size=34, color=YELLOW)\
                        .to_edge(UP)
        self.play(Write(heading))

        # -- Axes --
        ax = Axes(x_range=[-3.5, 3.5, 1], y_range=[-3.5, 3.5, 1],
                  x_length=5.5, y_length=5.5,
                  axis_config={"include_tip": True, "tip_length": 0.2,
                               "stroke_width": 1.5})\
             .to_edge(LEFT, buff=0.5).shift(DOWN*0.3)
        self.play(Create(ax))

        # -- Initial Z = I (unit disk + basis vectors) --
        def make_disk(transform=None):
            N = 80
            pts = []
            for i in range(N+1):
                t = i / N * TAU
                v = np.array([np.cos(t), np.sin(t)])
                if transform is not None:
                    v = transform @ v
                pts.append(ax.c2p(*v))
            return Polygon(*pts, color=GRAY, fill_color=GRAY, fill_opacity=0.4,
                            stroke_width=1.5)
        
        def make_grid(ax, M, x_range=(-2, 2), y_range=(-2, 2), step=0.5):
            lines = VGroup()

            # vertical lines: x = c
            for c in np.arange(x_range[0], x_range[1] + 1e-9, step):
                p1 = np.array([c, y_range[0]])
                p2 = np.array([c, y_range[1]])
                q1 = ax.c2p(*(M @ p1))
                q2 = ax.c2p(*(M @ p2))
                lines.add(Line(q1, q2, stroke_opacity=0.45, stroke_width=1))

            # horizontal lines: y = c
            for c in np.arange(y_range[0], y_range[1] + 1e-9, step):
                p1 = np.array([x_range[0], c])
                p2 = np.array([x_range[1], c])
                q1 = ax.c2p(*(M @ p1))
                q2 = ax.c2p(*(M @ p2))
                lines.add(Line(q1, q2, stroke_opacity=0.45, stroke_width=1))

            return lines

        def make_arrow(vec, color, transform=None):
            v = transform @ vec if transform is not None else vec
            return Arrow(ax.c2p(0, 0), ax.c2p(*v), buff=0,
                            color=color, stroke_width=3, max_tip_length_to_length_ratio=0.15)

        I_mat = np.eye(2)
        disk  = make_disk()
        grid = make_grid(ax, I_mat)
        arr_e1 = make_arrow(np.array([1,0]), RED)
        arr_e2 = make_arrow(np.array([0,1]), BLUE)

        Z_name = MathTex("I_{2}", font_size=30).next_to(ax, UP, buff=0.08)
        self.play(FadeIn(disk), FadeIn(grid), GrowArrow(arr_e1), GrowArrow(arr_e2), Write(Z_name))
        self.wait(0.5)

        # -- Corner panel: current matrix value --
        corner_bg = Rectangle(width=7, height=5.5, color=DARK_GRAY,
                                fill_color=DARK_GRAY, fill_opacity=0.5)\
                        .to_edge(RIGHT, buff=0.2).shift(DOWN*0.3)
        self.play(FadeIn(corner_bg))

        def corner_label(name_tex, mat, color=WHITE):
            lbl = MathTex(name_tex, font_size=28, color=color)
            m   = small_matrix(mat, color=color)
            grp = VGroup(lbl, m).arrange(DOWN, buff=0.15)\
                                .move_to(corner_bg.get_center())
            return grp

        eq_symbol = MathTex("=", font_size=28, color=PINK)
        # -- Step 0: show Z = I --
        crn = corner_label("I", I_mat)
        self.play(FadeIn(crn))
        self.wait(0.1)

        # -- Step 1: apply V^T --
        step1_title = mixed_tex("Step 1: multiply by $$V^T$$  (rotate)", font_size=24, color=RED)\
                            .next_to(heading, DOWN, buff=0.1)
        self.play(Write(step1_title))

        # Show V^T corner
        ZVt = Vt_mat
        crn_vtz = corner_label("V^T \\cdot I =", ZVt, RED)
        self.play(Transform(crn, crn_vtz))
        self.wait(0.2)

        # Transform disk
        disk2   = make_disk(Vt_mat)
        grid2 = make_grid(ax, Vt_mat)
        arr1_e1 = make_arrow(np.array([1,0]), RED,  Vt_mat)
        arr1_e2 = make_arrow(np.array([0,1]), BLUE, Vt_mat)
        self.play(Transform(disk, disk2),
                  Transform(grid, grid2),
                  Transform(arr_e1, arr1_e1),
                  Transform(arr_e2, arr1_e2), run_time=1.8)

        # Show result V^T·Z corner
        crn_vt = corner_label("V^T =", Vt_mat, RED)
        self.play(Transform(crn, crn_vt))
        self.wait(0.5)

        # -- Step 2: apply Σ --
        self.play(FadeOut(crn))

        T2 = Sigma_mat @ Vt_mat

        step2_title = mixed_tex("Step 2: multiply by $$\Sigma$$  (scale)", font_size=24, color=GREEN)\
                            .next_to(heading, DOWN, buff=0.1)
        self.play(Transform(step1_title, step2_title))

        res_tex = MathTex("\\Sigma \\cdot V^T =", color=GREEN, font_size=28)\
                            .move_to(corner_bg.get_top(), aligned_edge=UP)
        grp_mat = VGroup(small_matrix(Sigma_mat, color=GREEN), small_matrix(Vt_mat, color=GREEN))\
                            .arrange(RIGHT, buff=0.1)\
                            .next_to(res_tex, DOWN, buff=0.2)
        res_mat = VGroup(eq_symbol, small_matrix(T2, color=PURE_CYAN))\
                            .arrange(RIGHT, buff=0.1)\
                            .next_to(grp_mat, DOWN, buff=0.2)
        self.play(Write(res_tex), FadeIn(grp_mat))
        self.wait(0.2)
        self.play(FadeIn(res_mat))

        disk3   = make_disk(T2)
        grid3   = make_grid(ax, T2)
        arr2_e1 = make_arrow(np.array([1,0]), RED,  T2)
        arr2_e2 = make_arrow(np.array([0,1]), BLUE, T2)
        self.play(Transform(disk, disk3),
                  Transform(grid, grid3),
                  Transform(arr_e1, arr2_e1),
                  Transform(arr_e2, arr2_e2), run_time=1.8)
        self.wait(0.7)


        # -- Step 3: apply U --
        T3 = U_mat @ Sigma_mat @ Vt_mat

        step3_title = mixed_tex("Step 3: multiply by $$U$$  (last rotate)", font_size=24, color=BLUE)\
                          .next_to(heading, DOWN, buff=0.1)
        self.play(Transform(step1_title, step3_title))

        new_tex = MathTex("A = U \\cdot \\Sigma \\cdot V^T =", color=BLUE, font_size=28)\
                          .move_to(corner_bg.get_top(), aligned_edge=UP)
        new_grp_mat = VGroup(small_matrix(U_mat, color=BLUE), small_matrix(Sigma_mat, color=BLUE), small_matrix(Vt_mat, color=BLUE))\
                          .arrange(RIGHT, buff=0.1)\
                          .next_to(res_tex, DOWN, buff=0.2)
        new_res_mat = VGroup(eq_symbol, small_matrix(T3, color=PURE_CYAN))\
                            .arrange(RIGHT, buff=0.1)\
                            .next_to(grp_mat, DOWN, buff=0.2)
        self.play(
            Transform(res_tex, new_tex),
            Transform(grp_mat, new_grp_mat),
            Transform(res_mat, new_res_mat),
        )

        disk4   = make_disk(T3)
        grid4   = make_grid(ax, T3)
        arr3_e1 = make_arrow(np.array([1,0]), RED,  T3)
        arr3_e2 = make_arrow(np.array([0,1]), BLUE, T3)
        self.play(Transform(disk, disk4),
                  Transform(grid, grid4),
                  Transform(arr_e1, arr3_e1),
                  Transform(arr_e2, arr3_e2), run_time=1.8)
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 5 - Introduce Diagonalization section
# ---------------------------------------------
class S05_Diag_Intro(Scene):
    def construct(self):
        heading = Text("Diagonalization", font_size=46, color=YELLOW)\
                      .to_edge(UP)
        self.play(Write(heading))

        line1 = Text("Analyze matrix A:", font_size=30)
        line2 = MathTex(
            "A = P \\cdot D \\cdot P^{-1}",
            color=BLUE,
            font_size=30
        )
        intro = VGroup(line1, line2)\
            .arrange(DOWN, aligned_edge=UP, buff=0.3)\
            .next_to(heading, DOWN, buff=0.5)
        self.play(FadeIn(intro, shift=DOWN*0.3))

        A_label = MathTex(r"A = \begin{bmatrix}3&1\\1&2\end{bmatrix}",
                          font_size=44).next_to(intro, DOWN, buff=0.6)
        self.play(Write(A_label))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 6 - Eigenvalues and eigenvectors
# ---------------------------------------------
class S06_Eigen(Scene):
    def construct(self):
        # -- Phuong trinh dac trung --
        heading = Text("Characteristic Equation", font_size=34, color=YELLOW).to_edge(UP)
        self.play(Write(heading))

        # 1. Dang tong quat
        gen_eq = MathTex(r"\det(A - \lambda I) = 0", font_size=45).next_to(heading, DOWN, buff=0.1)
        self.play(Write(gen_eq))
        self.wait(1)

        # 2. Trien khai
        a11, a12 = A_MAT[0,0], A_MAT[0,1]
        a21, a22 = A_MAT[1,0], A_MAT[1,1]

        mat_var = MathTex(
            r"\det \begin{bmatrix} "
            r"a_{11} - \lambda & a_{12} \\ "
            r"a_{21} & a_{22} - \lambda "
            r"\end{bmatrix} = 0",
            font_size=40
        ).next_to(gen_eq, DOWN, buff=1)
        
        self.play(Transform(gen_eq, mat_var))
        self.wait(1)

        # 3. Thay so
        mat_num = MathTex(
            r"\det \begin{bmatrix} "
            rf"{a11:.0f} - \lambda & {a12:.0f} \\ "
            rf"{a21:.0f} & {a22:.0f} - \lambda "
            r"\end{bmatrix} = 0",
            font_size=40
        ).move_to(mat_var)

        self.play(Transform(gen_eq, mat_num))
        self.wait(1)

        poly = MathTex(
            rf"({a11:.0f} - \lambda)({a22:.0f} - \lambda) - ({a12:.0f} \cdot {a21:.0f}) = 0",
            font_size=36, color=YELLOW
        ).next_to(mat_num, DOWN, buff=1)

        # Ve duong cheo
        line_main = Line(mat_num.get_corner(UL), mat_num.get_corner(DR), color=GREEN).scale(0.7)
        line_sub  = Line(mat_num.get_corner(UR), mat_num.get_corner(DL), color=RED).scale(0.7)
        
        self.play(Create(line_main))
        self.play(Create(line_sub))
        self.play(Write(poly))
        self.wait(2)

        self.play(FadeOut(gen_eq, poly, line_main, line_sub))

        # -- Eigenvalue display --
        h1 = Text("Eigenvalues and EigenVectors of A", font_size=34, color=YELLOW).to_edge(UP)
        self.play(Transform(heading, h1))

        lam1_tex = MathTex(
            rf"\lambda_1 = {fmt(eigvals[0])}",
            font_size=38, color=RED
        )
        lam2_tex = MathTex(
            rf"\lambda_2 = {fmt(eigvals[1])}",
            font_size=38, color=BLUE
        )
        eigs_grp = VGroup(lam1_tex, lam2_tex).arrange(RIGHT, buff=2)\
                       .next_to(heading, DOWN, buff=1.5).to_edge(LEFT)
        self.play(Write(lam1_tex), Write(lam2_tex))
        self.wait(0.6)

        # -- Eigenvectors --
        ev1 = eigvecs[:, 0]
        ev2 = eigvecs[:, 1]
        vec1_tex = MathTex(
            rf"v_1 = \begin{{bmatrix}}{fmt(ev1[0])}\\{fmt(ev1[1])}\end{{bmatrix}}",
            font_size=38, color=RED
        ).next_to(lam1_tex, DOWN, buff=1.5)
        vec2_tex = MathTex(
            rf"v_2 = \begin{{bmatrix}}{fmt(ev2[0])}\\{fmt(ev2[1])}\end{{bmatrix}}",
            font_size=38, color=BLUE
        ).next_to(lam2_tex, DOWN, buff=1.5)

        self.play(FadeIn(vec1_tex, shift=LEFT*0.2))
        self.play(FadeIn(vec2_tex, shift=LEFT*0.2))
        self.wait(1)

        # -- Axes to visualise --
        ax = Axes(x_range=[-2,2,1], y_range=[-2,2,1],
                  x_length=4.5, y_length=4.5,
                  axis_config={"include_tip": True, "tip_length": 0.18})\
             .to_edge(RIGHT, buff=0.4).shift(DOWN*0.5)
        self.play(Create(ax))

        sc = 1.5
        a1 = Arrow(ax.c2p(0,0), ax.c2p(ev1[0]*sc, ev1[1]*sc),
                   buff=0, color=RED, stroke_width=4)
        a2 = Arrow(ax.c2p(0,0), ax.c2p(ev2[0]*sc, ev2[1]*sc),
                   buff=0, color=BLUE, stroke_width=4)
        l1 = MathTex("v_1", color=RED, font_size=26)\
                 .next_to(ax.c2p(ev1[0]*sc, ev1[1]*sc), RIGHT, buff=0.07)
        l2 = MathTex("v_2", color=BLUE, font_size=26)\
                 .next_to(ax.c2p(ev2[0]*sc, ev2[1]*sc), LEFT, buff=0.07)
        self.play(GrowArrow(a1), GrowArrow(a2), Write(l1), Write(l2))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 7 - Build P, P^{-1}, D ; show A = P D P^{-1}
# ---------------------------------------------
class S07_Diag_Build(Scene):
    def construct(self):
        heading = mixed_tex("Construct diagonalization $$A = P \cdot D \cdot P^{-1}$$",
                       font_size=30, color=YELLOW).to_edge(UP)
        self.play(Write(heading))

        # -- Helper: labeled matrix block --
        def named_block(name_tex, mat=None, color=WHITE, decimals=3):
            name = MathTex(name_tex, font_size=32, color=color).set_stroke(width=0)

            if mat is None:
                return name

            m    = small_matrix(mat, decimals=decimals, color=color)
            return VGroup(name, m).arrange(DOWN, buff=0.18)

        # -- Step 1: show P from eigenvectors --
        step1 = Text("Merge eigenvectors to form matrix P", 
                     font_size=22, color=GRAY).next_to(heading, DOWN, buff=0.1)
        self.play(Write(step1))

        P_block = MathTex("P = [v_1 \quad v_2]", color=ORANGE).move_to(ORIGIN)
        self.play(FadeIn(P_block, shift=UP*0.3))
        self.wait(0.5)
        eq_symbol = Text("=")
        expanded_P = small_matrix(P_mat, color=ORANGE)
        grp_eq_exp_P = VGroup(eq_symbol, expanded_P).arrange(RIGHT, buff=0.1).next_to(P_block, RIGHT, buff=0.1)
        self.play(FadeIn(grp_eq_exp_P))
        self.wait(0.5)
        self.play(FadeOut(grp_eq_exp_P))
        expanded_P = named_block("P", P_mat, ORANGE).move_to(ORIGIN)
        self.play(Transform(P_block, expanded_P))
        self.wait(1)

        # -- Step 2: show P^{-1} beside P --
        step2 = Text("Calculate P^{-1}", font_size=22, color=GRAY)\
                    .next_to(heading, DOWN, buff=0.1)
        self.play(Transform(step1, step2))

        Pinv_block = named_block("P^{-1}", P_inv, PURPLE)
        grp_P_Pinv = VGroup(P_block, Pinv_block).arrange(RIGHT, buff=1.2).move_to(ORIGIN)
        self.play(P_block.animate.move_to(grp_P_Pinv[0].get_center()),
                  FadeIn(Pinv_block, shift=LEFT*0.3))
        self.wait(1)

        # -- Step 3: insert D in the middle, prepend A= --
        self.play(FadeOut(step1))

        D_block  = named_block("D", None, GREEN)
        eq_A     = MathTex("A =", font_size=36)
        full_row = VGroup(eq_A, P_block.copy(), D_block, Pinv_block.copy())\
                       .arrange(RIGHT, buff=0.55).scale(0.9).move_to(ORIGIN)

        self.play(
            FadeOut(grp_P_Pinv),
            FadeIn(full_row, shift=UP*0.2)
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 8 - Show D = P^{-1} A P  with value
# ---------------------------------------------
class S08_Diag_D(Scene):
    def construct(self):
        heading = mixed_tex("Calculate matrix $$D$$", font_size=38, color=YELLOW).to_edge(UP)
        self.play(Write(heading))

        def sm(mat, color=WHITE, dec=3):
            return small_matrix(mat, decimals=dec, color=color)

        Pinv_m = sm(P_inv, PURPLE)
        A_m    = sm(A_MAT, WHITE, dec=1)
        P_m    = sm(P_mat, ORANGE)
        eq1    = MathTex("D", "=", "P^{-1}", "A", "P", font_size=44)
        eq1.set_color_by_tex("P^{-1}", PURPLE)
        eq1.set_color_by_tex("A", WHITE)
        eq1.set_color_by_tex("P", ORANGE)
        eq1.set_color_by_tex("D", GREEN)

        self.play(Write(eq1.next_to(heading, DOWN, buff=1)))
        self.wait(0.8)

        # Expand with values
        lhs = MathTex("D =", font_size=36, color=GREEN)
        row = VGroup(lhs, Pinv_m,
                     MathTex("\cdot", font_size=30), A_m,
                     MathTex("\cdot", font_size=30), P_m)\
              .arrange(RIGHT, buff=0.25).scale(0.9).next_to(eq1, DOWN, buff=0.55)
        self.play(FadeIn(row, shift=UP*0.2))
        self.wait(0.8)

        # Show = D value
        D_val = small_matrix(D_mat, color=GREEN)
        equals = MathTex("=", font_size=36)
        D_row  = VGroup(equals, D_val).arrange(RIGHT, buff=0.2)\
                     .next_to(row, DOWN, buff=0.4)
        self.play(Write(equals), FadeIn(D_val, shift=LEFT*0.2))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 9 - Final: A = P D P^{-1} with values
# ---------------------------------------------
class S09_Diag_Final(Scene):
    def construct(self):
        heading = Text("Result:", font_size=38, color=YELLOW)\
                      .to_edge(UP)
        self.play(Write(heading))

        def sm(mat, color=WHITE, dec=3):
            return small_matrix(mat, decimals=dec, color=color)

        # -- Symbolic equation with box --
        eq_sym = MathTex("A", "=", "P", "D", "P^{-1}", font_size=46)
        eq_sym.set_color_by_tex("P", ORANGE)
        eq_sym.set_color_by_tex("D", GREEN)
        eq_sym.set_color_by_tex("A", WHITE)
        box = SurroundingRectangle(eq_sym, color=YELLOW, buff=0.2)

        grp_sym = VGroup(eq_sym, box).next_to(heading, DOWN, buff=0.5)
        self.play(Write(eq_sym), Create(box))
        self.wait(1)

        # -- Expand A --
        A_lbl  = MathTex("A", font_size=34)
        A_m    = sm(A_MAT, WHITE, dec=1)
        A_row  = VGroup(A_lbl, A_m).arrange(DOWN, buff=0.1)

        eq_row_sym = MathTex("=", font_size=34)

        P_lbl  = MathTex("P", font_size=28, color=ORANGE)
        P_m    = sm(P_mat, ORANGE)
        D_lbl  = MathTex("D", font_size=28, color=GREEN)
        D_m    = sm(D_mat, GREEN)
        Pi_lbl = MathTex("P^{-1}", font_size=28, color=PURPLE)
        Pi_m   = sm(P_inv, PURPLE)

        P_blk  = VGroup(P_lbl,  P_m ).arrange(DOWN, buff=0.1)
        D_blk  = VGroup(D_lbl,  D_m ).arrange(DOWN, buff=0.1)
        Pi_blk = VGroup(Pi_lbl, Pi_m).arrange(DOWN, buff=0.1)

        full = VGroup(A_row, eq_row_sym, P_blk, D_blk, Pi_blk)\
                   .arrange(RIGHT, buff=0.4).scale(0.82)\
                   .next_to(grp_sym, DOWN, buff=0.55)
        self.play(FadeIn(full, shift=UP*0.3))
        self.wait(3)

        # Highlight
        self.play(Indicate(grp_sym, color=YELLOW, scale_factor=1.06))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------
# Scene 10 - End card
# ---------------------------------------------
class S10_End(Scene):
    def construct(self):
        thanks = Text("Thank you for watching!", font_size=56, color=YELLOW)
        sub    = Text("SVD  and  Diagonalization", font_size=30, color=GRAY)\
                     .next_to(thanks, DOWN, buff=0.5)
        self.play(Write(thanks), FadeIn(sub, shift=UP*0.2))
        self.wait(3)
        self.play(FadeOut(thanks), FadeOut(sub))


# ---------------------------------------------
# Concatenated single-render entry point
# manim -pqm manim_show.py FullShow   (720p)
# manim -pqh manim_show.py FullShow   (1080p)
# ---------------------------------------------
class demo_video(Scene):
    def construct(self):
        _scenes = [S01_Title, S02_SVD_Intro, S03_SVD_Decompose,
                   S04_SVD_Transform, S05_Diag_Intro, S06_Eigen,
                   S07_Diag_Build, S08_Diag_D, S09_Diag_Final, S10_End]
        for i, Cls in enumerate(_scenes):
            Cls.construct(self)
            if i < len(_scenes) - 1:
                if self.mobjects:
                    self.play(FadeOut(*self.mobjects), run_time=0.4)
                self.wait(0.2)