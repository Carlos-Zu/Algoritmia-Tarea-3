"""
Tarea 3 - Opcion B: ajuste de la temperatura inicial T0 del Recocido Simulado (2-opt sobre TSP)
con Metodologia de Superficie de Respuesta (diseño central compuesto de un factor).
Solo requiere numpy, scipy y matplotlib.
"""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from tsp_common import tour_length, random_tour, simulated_annealing, optimal_tour

# ---------------- Parametros del experimento ----------------
MATRICULA = 2154016
TAMANOS = [5, 10, 15]                 # bloque: tamaño del problema (una instancia por tamaño)
T_MIN, T_MAX = 1.0, 300.0             # rango de temperatura definido para el problema (ver reporte)
T_FINAL = 0.01                        # temperatura final del enfriamiento (fija para todos los niveles)
ITER_POR_CIUDAD = 10                  # presupuesto fijo: 10 * n iteraciones de recocido en cada corrida
REPS_EXTERNOS, REPS_CENTRO = 5, 10    # replicas: 5 en cada punto externo y 10 en el centro (30 por tamaño)
N_BOOT = 10_000                       # remuestreos bootstrap
ALFA = 0.05                           # nivel de significancia (IC del 95 %)

# ---------------- Niveles de temperatura (r se genera UNA sola vez) ----------------
random.seed(MATRICULA)
r = random.random()
NOMBRES = ["axial bajo", "bajo", "centro", "alto", "axial alto"]
FRACCIONES = [0.2 * r + k for k in (0.0, 0.2, 0.4, 0.6, 0.8)]
TEMPERATURAS = [T_MIN + f * (T_MAX - T_MIN) for f in FRACCIONES]


def esquema_geometrico(T0, n_iter, T_f=T_FINAL):
    """Enfriamiento exponencial de T0 a T_f en exactamente n_iter iteraciones (T = 0 despues)."""
    razon = (T_f / T0) ** (1 / n_iter)
    return lambda t: T0 * razon ** t if t <= n_iter else 0


def crear_instancia(n):
    rng = random.Random(MATRICULA + n)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]


def correr_experimento(n):
    """Devuelve listas (temperatura, costo) con las observaciones individuales de un tamaño n."""
    pts = crear_instancia(n)
    T_obs, costos = [], []
    for i, T0 in enumerate(TEMPERATURAS):
        reps = REPS_CENTRO if i == 2 else REPS_EXTERNOS
        for j in range(reps):
            random.seed(1_000_000 * n + 1_000 * i + j)            # semilla distinta en cada corrida
            tour, _, _ = simulated_annealing(pts, esquema_geometrico(T0, ITER_POR_CIUDAD * n), random_tour(n))
            T_obs.append(T0)
            costos.append(tour_length(pts, tour))
    return pts, np.array(T_obs), np.array(costos)


def bootstrap_mediana(y, rng):
    """Mediana bootstrap y su intervalo percentil del 95 %."""
    y = np.asarray(y)
    medianas = np.median(y[rng.integers(0, len(y), size=(N_BOOT, len(y)))], axis=1)
    return np.median(medianas), np.percentile(medianas, 100 * ALFA / 2), np.percentile(medianas, 100 * (1 - ALFA / 2))


def ajuste_cuadratico(T, y, T_centro=None):
    """Costo = b0 + b1*T + b2*T^2 por minimos cuadrados, con ANOVA de regresion y falta de ajuste."""
    T_centro = TEMPERATURAS[2] if T_centro is None else T_centro
    N = len(y)
    X = np.column_stack([np.ones(N), T, T ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    sse = float(np.sum((y - y_hat) ** 2))
    ssr = float(np.sum((y_hat - y.mean()) ** 2))
    gl_res = N - 3
    mse = sse / gl_res
    cov = mse * np.linalg.inv(X.T @ X)
    t_b = beta / np.sqrt(np.diag(cov))
    p_b = 2 * stats.t.sf(np.abs(t_b), gl_res)
    F_reg = (ssr / 2) / mse
    p_reg = stats.f.sf(F_reg, 2, gl_res)
    # error puro: variacion entre las replicas del punto central (no depende del modelo);
    # falta de ajuste: lo que resta del residual
    yc = y[T == T_centro]
    sspe = float(np.sum((yc - yc.mean()) ** 2))
    gl_pe = len(yc) - 1
    sslof = sse - sspe
    gl_lof = gl_res - gl_pe
    F_lof = (sslof / gl_lof) / (sspe / gl_pe)
    p_lof = stats.f.sf(F_lof, gl_lof, gl_pe)
    return dict(beta=beta, p_beta=p_b, X=X, mse=mse, cov=cov, gl_res=gl_res, ssr=ssr, sse=sse,
                F_reg=F_reg, p_reg=p_reg, sspe=sspe, gl_pe=gl_pe, sslof=sslof, gl_lof=gl_lof,
                F_lof=F_lof, p_lof=p_lof, sst=ssr + sse)


if __name__ == "__main__":
    rng_boot = np.random.default_rng(MATRICULA)
    print(f"Matrícula (semilla) = {MATRICULA}   r = {r:.6f}   rango de temperatura = [{T_MIN}, {T_MAX}]")
    print(f"{'nivel':12s} {'fracción':>9} {'T0':>8}")
    for nombre, f, T0 in zip(NOMBRES, FRACCIONES, TEMPERATURAS):
        print(f"{nombre:12s} {f:>9.4f} {T0:>8.2f}")
    print(f"Presupuesto: {ITER_POR_CIUDAD}·n iteraciones, T_final = {T_FINAL}; réplicas: {REPS_EXTERNOS} por punto externo y {REPS_CENTRO} en el centro")

    resumen = []
    for n in TAMANOS:
        pts, T_obs, y = correr_experimento(n)
        niveles = np.array(TEMPERATURAS)
        boot = [bootstrap_mediana(y[T_obs == T0], rng_boot) for T0 in niveles]

        print(f"\n================ n = {n} ({len(y)} corridas) ================")
        print(f"{'nivel':12s} {'T0':>8} {'réplicas':>9} {'media':>9} {'mediana boot.':>14} {'IC 95 % (bootstrap)':>24}")
        for nombre, T0, (med, lo, hi) in zip(NOMBRES, niveles, boot):
            yy = y[T_obs == T0]
            print(f"{nombre:12s} {T0:>8.2f} {len(yy):>9d} {yy.mean():>9.2f} {med:>14.2f} {f'[{lo:.2f}, {hi:.2f}]':>24}")

        if np.ptp(y) == 0:
            opt = tour_length(pts, optimal_tour(pts)) if n <= 9 else None
            print("Todas las corridas dieron exactamente el mismo costo: no hay variación que explicar, "
                  "el ajuste cuadrático y el ANOVA no están definidos.")
            if opt is not None:
                print(f"Costo óptimo (fuerza bruta) = {opt:.2f}; costo obtenido = {y[0]:.2f}")
            resumen.append((n, None, None, boot[2], None, None, None))
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.errorbar(niveles, [b[0] for b in boot], yerr=[[b[0] - b[1] for b in boot], [b[2] - b[0] for b in boot]],
                        fmt="s", color="tab:red", capsize=4, label="mediana bootstrap e IC 95 %")
            ax.scatter(T_obs, y, alpha=.5, s=22, label="corridas observadas")
            ax.set_title(f"n = {n}: sin variación entre corridas (costo constante)")
            ax.set_xlabel("temperatura inicial T0"); ax.set_ylabel("costo (longitud del tour)")
            ax.legend(); ax.grid(alpha=.3)
            plt.tight_layout(); plt.savefig(f"tarea3_curva_n{n}.png", dpi=130); plt.close()
            continue

        m = ajuste_cuadratico(T_obs, y)
        b0, b1, b2 = m["beta"]
        N = len(y)
        print(f"\nModelo: costo = {b0:.3f} + ({b1:.4f})·T + ({b2:.6f})·T²")
        print(f"{'coeficiente':12s} {'valor':>12} {'p-value':>10}")
        for nom, b, p in zip(["β0", "β1", "β2"], m["beta"], m["p_beta"]):
            print(f"{nom:12s} {b:>12.6f} {p:>10.4f}")
        print("\nANOVA de regresión")
        print(f"{'fuente':16s} {'gl':>4} {'SC':>12} {'MC':>12} {'F':>8} {'p':>8}")
        print(f"{'Regresión':16s} {2:>4d} {m['ssr']:>12.2f} {m['ssr'] / 2:>12.2f} {m['F_reg']:>8.3f} {m['p_reg']:>8.4f}")
        print(f"{'Residual':16s} {m['gl_res']:>4d} {m['sse']:>12.2f} {m['sse'] / m['gl_res']:>12.2f}")
        print(f"{'  Falta de ajuste':16s} {m['gl_lof']:>4d} {m['sslof']:>12.2f} {m['sslof'] / m['gl_lof']:>12.2f} {m['F_lof']:>8.3f} {m['p_lof']:>8.4f}")
        print(f"{'  Error puro':16s} {m['gl_pe']:>4d} {m['sspe']:>12.2f} {m['sspe'] / m['gl_pe']:>12.2f}")
        print(f"{'Total':16s} {N - 1:>4d} {m['sst']:>12.2f}")

        t_lo, t_hi = niveles.min(), niveles.max()
        if b2 > 0:
            T_star = -b1 / (2 * b2)
            dentro = t_lo <= T_star <= t_hi
            print(f"\nβ2 > 0 (parábola hacia arriba): T* = -β1/(2β2) = {T_star:.2f}  "
                  f"({'dentro' if dentro else 'FUERA'} del rango experimental [{t_lo:.2f}, {t_hi:.2f}])")
            T_sel = T_star if dentro else (t_lo if (b0 + b1 * t_lo + b2 * t_lo ** 2) < (b0 + b1 * t_hi + b2 * t_hi ** 2) else t_hi)
        else:
            T_star = None
            T_sel = t_lo if (b0 + b1 * t_lo + b2 * t_lo ** 2) < (b0 + b1 * t_hi + b2 * t_hi ** 2) else t_hi
            print(f"\nβ2 <= 0: la parábola no tiene mínimo; el menor costo ajustado está en el extremo T = {T_sel:.2f}")
        j = int(np.argmin(np.abs(niveles - T_sel)))
        resumen.append((n, T_sel, T_star, boot[j], j, m["p_reg"], m["p_beta"][2]))

        # Grafica: curva ajustada, banda de confianza del 95 % de la media, observaciones y mediana bootstrap con IC
        Tg = np.linspace(t_lo, t_hi, 200)
        Xg = np.column_stack([np.ones_like(Tg), Tg, Tg ** 2])
        yg = Xg @ m["beta"]
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, m["cov"], Xg))
        tc = stats.t.ppf(1 - ALFA / 2, m["gl_res"])
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.fill_between(Tg, yg - tc * se, yg + tc * se, color="tab:blue", alpha=.18, label="IC 95 % de la curva ajustada")
        ax.plot(Tg, yg, color="tab:blue", label="curva cuadrática ajustada")
        jit = (np.random.default_rng(1).random(len(T_obs)) - .5) * 0.02 * (T_MAX - T_MIN)
        ax.scatter(T_obs + jit, y, alpha=.45, s=22, color="gray", label="corridas observadas")
        ax.errorbar(niveles, [b[0] for b in boot], yerr=[[b[0] - b[1] for b in boot], [b[2] - b[0] for b in boot]],
                    fmt="s", color="tab:red", capsize=4, label="mediana bootstrap e IC 95 %")
        if T_star is not None and t_lo <= T_star <= t_hi:
            ax.axvline(T_star, color="green", ls="--", label=f"T* = {T_star:.1f}")
        ax.set_xlabel("temperatura inicial T0"); ax.set_ylabel("costo (longitud del tour)")
        ax.set_title(f"n = {n}: costo vs temperatura (p regresión = {m['p_reg']:.3f}, p β2 = {m['p_beta'][2]:.3f})")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"tarea3_curva_n{n}.png", dpi=130); plt.close()

    print("\n================ Tabla resumen ================")
    print(f"{'n':>3} {'T seleccionada':>15} {'nivel más cercano':>18} {'mediana bootstrap':>18} {'IC 95 %':>20} {'p regresión':>12} {'p β2':>8}")
    for n, T_sel, T_star, b, j, p_reg, p_b2 in resumen:
        med, lo, hi = b
        if T_sel is None:
            print(f"{n:>3} {'indeterminada':>15} {NOMBRES[2] + f' ({TEMPERATURAS[2]:.2f})':>18} {med:>18.2f} {f'[{lo:.2f}, {hi:.2f}]':>20} {'-':>12} {'-':>8}")
        else:
            print(f"{n:>3} {T_sel:>15.2f} {NOMBRES[j] + f' ({TEMPERATURAS[j]:.2f})':>18} {med:>18.2f} {f'[{lo:.2f}, {hi:.2f}]':>20} {p_reg:>12.4f} {p_b2:>8.4f}")
    print("\nGráficas guardadas: " + ", ".join(f"tarea3_curva_n{n}.png" for n in TAMANOS))
