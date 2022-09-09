import jax
import jax.numpy as jnp
from jax import custom_jvp, custom_vjp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
import sys


@custom_vjp
def svd(A):
    # XXX
    # return jnp.linalg.svd(A, full_matrices=False)
    return jnp.linalg.svd(A, full_matrices=True)


def _safe_reciprocal(x, epsilon=1e-20):
    return x / (x * x + epsilon)


def h(x):
    return jnp.conj(jnp.transpose(x))


def jaxsvd_fwd(A):
    u, s, v = svd(A)
    return (u, s, v), (u, s, v)


def jaxsvd_bwd(r, tangents):
    U, S, V = r
    du, ds, dv = tangents

    dU = jnp.conj(du)
    dS = jnp.conj(ds)
    dV = jnp.transpose(dv)

    # XXX check dimensions
    m = U.shape[0]
    n = V.shape[1]
    assert(m <= n)

    # XXX shorten Vh and dVh
    full_V = V
    full_dV = dV
    V = V[:m,:]
    dV = dV[:,:m]

    # XXX The additional n-m vectors are used in the 
    # full factorization
    Vr = full_V[m:,:]
    dVr = full_dV[:,m:]

    # XXX pad dS
    dS = jnp.diag(dS)
    if m < n:
        padding = jnp.zeros((m, n - m))
        dS = jnp.concatenate((dS, padding), axis=1)

    ms = jnp.diag(S)
    ms1 = jnp.diag(_safe_reciprocal(S))
    # XXX
    # dAs = U @ jnp.diag(dS) @ V
    dAs = U @ dS @ full_V

    F = S * S - (S * S)[:, None]
    F = _safe_reciprocal(F) - jnp.diag(jnp.diag(_safe_reciprocal(F)))

    J = F * (h(U) @ dU)
    dAu = U @ (J + h(J)) @ ms @ V

    K = F * (V @ dV)
    dAv = U @ ms @ (K + h(K)) @ V

    # XXX Only apply additional term if input is complex
    if jnp.iscomplex(U).any() or jnp.iscomplex(V).any():
        O = h(dU) @ U @ ms1
        dAc = -1 / 2.0 * U @ (jnp.diag(jnp.diag(O - jnp.conj(O)))) @ V
    else:
        dAc = 0

    dAv = dAv + U @ ms1 @ h(dV) @ (jnp.eye(jnp.size(V[1, :])) - h(V) @ V)
    # XXX Extra term for full SVD case
    dAv = dAv - U @ ms1 @ V @ dVr @ Vr
    dAu = dAu + (jnp.eye(jnp.size(U[:, 1])) - U @ h(U)) @ dU @ ms1 @ V
    grad_a = jnp.conj(dAv + dAu + dAs + dAc)
    return (grad_a,)


svd.defvjp(jaxsvd_fwd, jaxsvd_bwd)


def l(A):
    u, s, v = svd(A)
    return jnp.real(u[0, -1] * v[-1, 0])


def test(m, n):
    Ax = np.random.randn(m, n)
    Ay = np.random.randn(m, n)
    A = jnp.array(Ax + 1.0j * Ay).astype(jnp.complex128)
    print("input:\n", A)
    # auto diff
    DA_ad = jax.grad(l)(A)

    print("auto diff:\n", DA_ad)
    # numerical
    d = 1e-6
    DA = np.zeros(shape=(m, n), dtype=np.complex128)
    for i in range(0, m):
        for j in range(0, n):
            dA = np.zeros(shape=(m, n))
            dA[i, j] = 1
            DA[i, j] = (l(A + d * dA) - l(A)) / d - 1.0j * (
                l(A + d * 1.0j * dA) - l(A)
            ) / d
    print("numerical:\n", DA)
    # difference
    # print("difference:\n",DA-DA_ad)
    # XXX
    # print("close?:\n", np.allclose(DA, DA_ad))
    close = np.allclose(DA, DA_ad)
    print("close?:\n", close)
    return close

# XXX test real inputs
def test_real(m, n):
    A = np.random.randn(m, n)

    print("input:\n", A)
    # auto diff
    DA_ad = jax.grad(l)(A)

    print("auto diff:\n", DA_ad)
    # numerical
    d = 1e-6
    DA = np.zeros(shape=(m, n))
    for i in range(0, m):
        for j in range(0, n):
            dA = np.zeros(shape=(m, n))
            dA[i, j] = 1
            DA[i, j] = (l(A + d * dA) - l(A)) / d 
    print("numerical:\n", DA)
    # NOTE did have to add an atol
    close = np.allclose(DA, DA_ad, atol=1e-6)
    if not close:
        print("difference:\n",DA-DA_ad)
    print("close?:\n", close)
    return close


# XXX
# if __name__ == "__main__":
#     test(int(sys.argv[1]), int(sys.argv[2]))


print("3x3 complex input matrix. The full factorization is the same as the partial case, and numerical finite difference _is_ the same.")
assert(test(3, 3))
print("\n")

print("3x4 complex input matrix. When adding the additional term to the gradient, the numerical finite difference _is not_ the same.")
assert(not test(3, 4))
print("\n")

print("As a sanity check, let's check that the additional term is correct for the real input full SVD case.")
print("\n")

print("3x3 real input matrix. The numerical finite difference _is_ the same")
assert(test_real(3, 3))
print("\n")

print("3x4 real input matrix. The numerical finite difference _is_ the same")
assert(test_real(3, 4))
print("\n")
