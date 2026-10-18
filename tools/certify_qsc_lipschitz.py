"""Exact-rational, full-domain certificate for a QSC inverse Lipschitz bound.

This encloses whole intervals, rather than probing derivatives at sampled
points. It uses a rational Machin-series enclosure of pi, square checks for
1/sqrt(2), and alternating Taylor bounds for sine and cosine below 11/42 radians.
No NumPy, floating-point trigonometry, or optional packages are needed.

Rotate/reflect each QSC sector to u=p, v=p*t, with 0<=p,t<=1. Put a=pi/12,
b=1/sqrt(2), z=1-b*cos(a*t), D=2-1/(2*z), B=D'(t), and H^2=D*(2-p^2*D).
The exact inverse-map Jacobian in orthonormal spherical coordinates is

    [(2*D-t*B)/H, B/H]
    [-t*H*a/D,    H*a/D].

Its determinant is 2*a=pi/6. Its squared Frobenius norm is

    Q/H^2 + (1+t^2)*H^2*a^2/D^2,  Q=(2*D-t*B)^2+B^2.

For fixed t this is convex in H^2, so its maximum over p occurs at p=0 or
p=1. Put B=41/50 and D=(pi_lower/6)^2. The rational enclosures below prove
both endpoint traces <B+D/B on every closed t interval. If lambda1,lambda2
are the squared singular values, their product is (pi/6)^2>D and <B^2. Therefore
(B-lambda1)*(B-lambda2)>0, and both must be <B. Sector continuity and
integration along a face segment extend this to a global squared Lipschitz
bound of B, including sector boundaries and the center.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path


def _pi_bounds() -> tuple[F, F]:
    """Enclose pi by Machin's identity and four alternating arctan terms."""
    def atan_bounds(reciprocal):
        x = F(1, reciprocal)
        lower = sum((-1)**k * x**(2*k + 1) / (2*k + 1) for k in range(4))
        return lower, lower + x**9 / 9

    lower5, upper5 = atan_bounds(5)
    lower239, upper239 = atan_bounds(239)
    # tan(4*atan(1/5)-atan(1/239))=1, with the angle in (0,3/2)<(0,pi/2).
    twice = 2 * F(1, 5) / (1 - F(1, 5)**2)
    four = 2 * twice / (1 - twice**2)
    assert (four - F(1, 239)) / (1 + four * F(1, 239)) == 1
    assert 0 < 4 * lower5 - upper239 < 4 * upper5 - lower239 < F(3, 2)
    return 16 * lower5 - 4 * upper239, 16 * upper5 - 4 * lower239


DEFAULT_SUBDIVISIONS = 1024
INVERSE_LIPSCHITZ_SQUARED = F(41, 50)
PI_LOWER, PI_UPPER = _pi_bounds()
DETERMINANT_SQUARED_LOWER = (PI_LOWER / 6)**2
TRACE_LIMIT = INVERSE_LIPSCHITZ_SQUARED + DETERMINANT_SQUARED_LOWER / INVERSE_LIPSCHITZ_SQUARED
INV_SQRT2_LOWER, INV_SQRT2_UPPER = F(707106, 1000000), F(707107, 1000000)


def _sin_lower(x: F) -> F:
    return x - x**3 / 6


def _sin_upper(x: F) -> F:
    return x - x**3 / 6 + x**5 / 120


def _cos_lower(x: F) -> F:
    return 1 - x**2 / 2 + x**4 / 24 - x**6 / 720


def _cos_upper(x: F) -> F:
    return 1 - x**2 / 2 + x**4 / 24


def certify(subdivisions: int = DEFAULT_SUBDIVISIONS) -> tuple[dict, ...]:
    """Return exact interval bounds, raising if this partition cannot certify.

    A failed coarse partition means only that its enclosure is too loose.
    Every comparison and bound contributing to acceptance is rational.
    """
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions <= 0:
        raise ValueError('subdivisions must be a positive integer')
    a_lower, a_upper = PI_LOWER / 12, PI_UPPER / 12
    b_lower, b_upper = INV_SQRT2_LOWER, INV_SQRT2_UPPER
    if not (0 < b_lower**2 < F(1, 2) < b_upper**2 < 1):
        raise AssertionError('invalid inverse-square-root enclosure')
    if not (0 < a_lower < a_upper < 1):
        raise AssertionError('Taylor interval is outside its decreasing-term range')
    if not (F(1, 4) < DETERMINANT_SQUARED_LOWER == (2 * a_lower)**2
            < (2 * a_upper)**2 < INVERSE_LIPSCHITZ_SQUARED**2):
        raise AssertionError('determinant enclosure does not prove the singular-value bound')
    records = []
    for index in range(subdivisions):
        lower, upper = F(index, subdivisions), F(index + 1, subdivisions)
        k_lower, k_upper = a_lower * lower, a_upper * upper
        z_lower = 1 - b_upper * _cos_upper(k_lower)
        z_upper = 1 - b_lower * _cos_lower(k_upper)
        d_lower = 2 - 1 / (2 * z_lower)
        d_upper = 2 - 1 / (2 * z_upper)
        b_derivative_lower = a_lower * b_lower * _sin_lower(k_lower) / (2 * z_upper**2)
        b_derivative_upper = a_upper * b_upper * _sin_upper(k_upper) / (2 * z_lower**2)
        if not (0 < z_lower <= z_upper and 0 < d_lower <= d_upper < 1
                and 0 <= b_derivative_lower <= b_derivative_upper
                and 2 * d_lower - upper * b_derivative_upper > 0):
            raise AssertionError(f'invalid positive factor enclosure at interval {index}')
        q_upper = (2 * d_upper - lower * b_derivative_lower)**2 + b_derivative_upper**2
        trace_p0 = q_upper / (2 * d_lower) + 2 * a_upper**2 * (1 + upper**2) / d_lower
        trace_p1 = (q_upper / (d_lower * (2 - d_lower))
                    + a_upper**2 * (1 + upper**2) * (2 - d_lower) / d_lower)
        trace_upper = max(trace_p0, trace_p1)
        if not trace_upper < TRACE_LIMIT:
            raise AssertionError(f'trace enclosure does not prove the bound at interval {index}')
        records.append(dict(index=index, t_lower=lower, t_upper=upper,
                            trace_p0_upper=trace_p0, trace_p1_upper=trace_p1,
                            trace_upper=trace_upper))
    return tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subdivisions', type=int, default=DEFAULT_SUBDIVISIONS)
    parser.add_argument('--output', type=Path, help='optional JSON certificate destination')
    args = parser.parse_args()
    records = certify(args.subdivisions)
    worst = max(records, key=lambda record: record['trace_upper'])
    summary = dict(certified=True,
                   inverse_lipschitz_bound=math.nextafter(math.sqrt(float(INVERSE_LIPSCHITZ_SQUARED)), math.inf),
                   inverse_lipschitz_squared_bound=str(INVERSE_LIPSCHITZ_SQUARED),
                   subdivisions=len(records), trace_upper=float(worst['trace_upper']),
                   trace_upper_exact=str(worst['trace_upper']), trace_limit=str(TRACE_LIMIT),
                   pi_lower=str(PI_LOWER), pi_upper=str(PI_UPPER),
                   determinant_squared_lower=str(DETERMINANT_SQUARED_LOWER),
                   worst_interval=worst['index'], arithmetic='exact rational interval enclosure')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(summary, intervals=[{key: str(value) if isinstance(value, F) else value
                                            for key, value in row.items()} for row in records])
        args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
