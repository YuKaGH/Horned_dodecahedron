import numpy as np, math
from scipy.spatial import ConvexHull

def gen(n, t=2.0):
    d = {}
    for m in range(1, 2**(n+1)-1):
        S = {i for i in range(n+1) if (m>>i)&1}; k = len(S)
        a = 1.0/(k + t*(n+1-k))
        d[m] = np.array([a if i in S else t*a for i in range(n+1)])
    return d

def mask(S):
    r = 0
    for i in S: r |= 1<<i
    return r

def proj(b): return b[:-1]          # барицентр -> R^n

def simplex_vol(pts):
    p = np.array(pts); n = p.shape[1]
    return abs(np.linalg.det(p[1:]-p[0]))/math.factorial(n)

def compute(n, t=2.0):
    d = gen(n, t); ALL = set(range(n+1))
    pts = np.array([proj(b) for b in d.values()])
    Vc = ConvexHull(pts).volume
    Vpyr = 0.0; nf = 0
    for i in range(n+1):
        for j in range(n+1):
            if i == j: continue
            Fm = [m for m in d if (m>>i)&1 and not (m>>j)&1]
            p1,p2 = d[mask({i})], d[mask({i,j})]
            p3,p4 = d[mask(ALL-{j})], d[mask(set(sorted(ALL-{i,j})))]
            A = np.column_stack([p2-p1, -(p4-p3)])
            s = np.linalg.lstsq(A, p3-p1, rcond=None)[0][0]
            apex = proj(p1 + s*(p2-p1))
            F = np.array([proj(d[m]) for m in Fm]); c = F.mean(0)
            _,_,Vt = np.linalg.svd(F-c, full_matrices=False)
            F2 = (F-c) @ Vt[:n-1].T                     # в (n-1)-мерное
            for simp in ConvexHull(F2).simplices:
                Vpyr += abs(np.linalg.det(F[simp]-apex))/math.factorial(n)
            nf += 1
    e = np.eye(n+1); sp = np.array([proj(e[k]) for k in range(n+1)])
    Vs = simplex_vol(sp)
    return dict(facets=nf, central=Vc/Vs, total=(Vc+Vpyr)/Vs)

for n in (2, 3, 4):
    print(n, compute(n))