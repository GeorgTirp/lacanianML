"""
Tier-1 v0: topological anti-collapse, minimal toy test.

Data: circle S^1 smoothly embedded in R^20 (3 random harmonics) + noise.
JEPA task: from encoding of x(alpha), predict encoding of x(alpha+delta),
predictor conditioned on delta. No stop-grad, no EMA -> collapse-prone.

Models:
  m0: plain JEPA, R^2 latent.                      (expected: collapse)
  m1: m0 + VICReg-style variance hinge.            (statistical anti-collapse)
  m2: phase head (u,v)->S^1, circular JEPA loss
      + equivariance loss (installs winding=1)
      + norm barrier (guards the gate), activated
        once winding ~ 1 on the probe loop.        (topological anti-collapse)

Kill switch at step STEPS1: m1 drops the variance term, m2 drops the
equivariance term (barrier stays). Phase 2 runs predictive loss only.

Logged on a fixed ordered noiseless probe loop (256 angles):
  sd    : mean per-dim std of raw latent  (collapse metric)
  Rbar  : |mean e^{i phi}| circular concentration (1 = phase collapse)
  W     : winding number of phi along the probe loop (wrapped-diff sum / 2pi)
  min_r : min raw norm ||(u,v)|| along the loop (distance to the gate)
"""
import argparse, math, json
import numpy as onp
import autograd.numpy as np
from autograd import grad

P = dict(D=20, K=3, noise=0.05, H=48, HP=32, batch=96, lr=2e-3,
         steps1=2500, steps2=4500, steps3=2500, every=20, margin=0.5,
         lam_e=5.0, lam_b=10.0, lam_v=1.0, dmax=math.pi)

# ---------- data ----------
rng_data = onp.random.default_rng(0)
A = rng_data.normal(size=(P['K'], P['D'])) / math.sqrt(P['K'])
B = rng_data.normal(size=(P['K'], P['D'])) / math.sqrt(P['K'])

def embed(alpha):
    ks = onp.arange(1, P['K'] + 1)[:, None]
    c = onp.cos(ks * alpha[None, :]); s = onp.sin(ks * alpha[None, :])
    return c.T @ A + s.T @ B

PROBE_ALPHA = onp.linspace(0, 2 * math.pi, 256, endpoint=False)
PROBE_X = embed(PROBE_ALPHA)

# ---------- model ----------
def init_mlp(sizes, rng):
    ps = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        ps.append(rng.normal(size=(i, o)) * math.sqrt(2.0 / i))
        ps.append(onp.zeros(o))
    return ps

def mlp(ps, x):
    h = x; n = len(ps) // 2
    for i in range(n):
        h = h @ ps[2 * i] + ps[2 * i + 1]
        if i < n - 1: h = np.tanh(h)
    return h

def normalize(z):
    return z / (np.sqrt(np.sum(z * z, axis=1, keepdims=True)) + 1e-8)

NE = 6  # encoder param arrays: [D,H,H,2] -> 3 layers

def loss_fn(params, xb, xb2, delta, model, use_aux, barrier_on, lam_c):
    enc_p, pred_p = params[:NE], params[NE:]
    z = mlp(enc_p, xb); z2 = mlp(enc_p, xb2)
    if lam_c > 0:  # explicit collapse pressure: pull latents to the origin
        pass  # added at the end so it applies to all models
    dfeat = onp.stack([onp.cos(delta), onp.sin(delta)], axis=1)
    if model == 'm2':
        zn, z2n = normalize(z), normalize(z2)
        pin = np.concatenate([zn, dfeat], axis=1)
        p = normalize(mlp(pred_p, pin))
        L = np.mean(1.0 - np.sum(p * z2n, axis=1))
        if use_aux:
            phi1 = np.arctan2(zn[:, 1], zn[:, 0])
            phi2 = np.arctan2(z2n[:, 1], z2n[:, 0])
            L = L + P['lam_e'] * np.mean(1.0 - np.cos(phi2 - phi1 - delta))
        if barrier_on:
            r = np.sqrt(np.sum(z * z, axis=1) + 1e-12)
            L = L + P['lam_b'] * np.mean(np.maximum(0.0, P['margin'] - r) ** 2)
    else:
        pin = np.concatenate([z, dfeat], axis=1)
        p = mlp(pred_p, pin)
        L = np.mean(np.sum((p - z2) ** 2, axis=1))
        if model == 'm1' and use_aux:
            mu = np.mean(z, axis=0)
            sd = np.sqrt(np.mean((z - mu) ** 2, axis=0) + 1e-8)
            L = L + P['lam_v'] * np.mean(np.maximum(0.0, 1.0 - sd))
    if lam_c > 0:
        L = L + lam_c * np.mean(np.sum(z * z, axis=1))
    return L

loss_grad = grad(loss_fn)

def wrap(d):
    return onp.mod(d + math.pi, 2 * math.pi) - math.pi

def evaluate(enc_p):
    z = mlp(enc_p, PROBE_X)
    z = onp.asarray(z)
    sd = float(onp.mean(onp.std(z, axis=0)))
    r = onp.sqrt(onp.sum(z * z, axis=1))
    phi = onp.arctan2(z[:, 1], z[:, 0])
    d = wrap(onp.diff(phi))
    d = onp.append(d, wrap(phi[0] - phi[-1]))
    W = float(onp.sum(d) / (2 * math.pi))
    Rbar = float(onp.abs(onp.mean(onp.exp(1j * phi))))
    return sd, Rbar, W, float(r.min())

# ---------- training ----------
def train(model, seed, lam_c3):
    rng = onp.random.default_rng(seed)
    enc = init_mlp([P['D'], P['H'], P['H'], 2], rng)
    pred = init_mlp([4, P['HP'], P['HP'], 2], rng)
    params = enc + pred
    m = [onp.zeros_like(p) for p in params]
    v = [onp.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    barrier_on = False
    logs = {k: [] for k in ['step', 'sd', 'Rbar', 'W', 'minr', 'barrier']}
    T = P['steps1'] + P['steps2'] + P['steps3']
    for t in range(T):
        use_aux = t < P['steps1']
        lam_c = lam_c3 if t >= P['steps1'] + P['steps2'] else 0.0
        alpha = rng.uniform(0, 2 * math.pi, P['batch'])
        delta = rng.uniform(-P['dmax'], P['dmax'], P['batch'])
        xb = embed(alpha) + rng.normal(size=(P['batch'], P['D'])) * P['noise']
        xb2 = embed(alpha + delta) + rng.normal(size=(P['batch'], P['D'])) * P['noise']
        g = loss_grad(params, xb, xb2, delta, model, use_aux, barrier_on, lam_c)
        for i in range(len(params)):
            m[i] = b1 * m[i] + (1 - b1) * g[i]
            v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
            mh = m[i] / (1 - b1 ** (t + 1)); vh = v[i] / (1 - b2 ** (t + 1))
            params[i] = params[i] - P['lr'] * mh / (onp.sqrt(vh) + eps)
        if t % P['every'] == 0 or t == T - 1:
            sd, Rbar, W, minr = evaluate(params[:NE])
            if model == 'm2' and not barrier_on and abs(W - 1.0) < 0.1:
                barrier_on = True
            logs['step'].append(t); logs['sd'].append(sd)
            logs['Rbar'].append(Rbar); logs['W'].append(W)
            logs['minr'].append(minr); logs['barrier'].append(int(barrier_on))
    return logs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['m0', 'm1', 'm2'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lamc', type=float, default=0.5)
    ap.add_argument('--tag', type=str, default='')
    a = ap.parse_args()
    logs = train(a.model, a.seed, a.lamc)
    onp.savez(f'logs_{a.model}_s{a.seed}{a.tag}.npz',
              **{k: onp.array(v) for k, v in logs.items()})
    i = -1
    print(json.dumps({k: round(float(logs[k][i]), 4) for k in ['sd', 'Rbar', 'W', 'minr']},
                     indent=None), '| final metrics', a.model)