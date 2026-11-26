"""Additional checks; run beside the submitted files with --project-dir PATH.

The course's check.py remains the authoritative numerical reference check.
These checks cover independent shapes, edge cases and generated inputs.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import numpy_homework as homework


def raises_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError('Expected ValueError')


def check_exercises():
    assert np.isclose(homework.angle_deg([1, 0], [0, 1]), 90)
    assert np.isclose(homework.angle_deg([1, 0], [-1, 0]), 180)
    assert np.isclose(homework.angle_deg([1, 0], [2, 0]), 0)
    raises_value_error(homework.angle_deg, [0, 0], [1, 0])
    raises_value_error(homework.angle_deg, [1, 0], [0, 0])
    a = np.array([3., 4.])
    saved = a.copy()
    np.testing.assert_allclose(homework.unit_vector(a), [.6, .8])
    np.testing.assert_array_equal(a, saved)
    raises_value_error(homework.unit_vector, [0, 0])
    query = np.array([1., 0.])
    rows = np.array([[1., 0.], [2., 0.], [-1., 0.], [0., 0.], [0., 1.]])
    np.testing.assert_allclose(homework.angles_to(query, rows), [0, 0, 180, np.nan, 90], equal_nan=True)
    assert homework.angles_to(query, np.empty((0, 2))).shape == (0,)
    raises_value_error(homework.angles_to, [0, 0], rows)
    indices, scores = homework.top_k_cosine(query, rows, np.int64(4))
    np.testing.assert_array_equal(indices, [0, 1, 4, 2])
    np.testing.assert_allclose(scores, [1, 1, 0, -1])
    for k in (0, -1, 5, 1.5, True):
        raises_value_error(homework.top_k_cosine, query, rows, k)
    raises_value_error(homework.top_k_cosine, [0, 0], rows, 1)
    raises_value_error(homework.top_k_cosine, query, np.zeros((2, 2)), 1)
    rng = np.random.default_rng(20261126)
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    distances, nearest = homework.pairwise_distances(A, B)
    expected = np.array([[np.linalg.norm(a - b) for b in B] for a in A])
    np.testing.assert_allclose(distances, expected)
    np.testing.assert_array_equal(nearest, expected.argmin(axis=1))
    X = np.array([[.8, -.6, .7], [-.4, -.5, .9]])
    W1 = np.array([[-.1, .2, -.8], [.3, -.5, .1]])
    W2 = np.array([[.2, -.6], [-.8, .52]])
    Z, H, Y = homework.small_forward(X, W1, np.array([-.15, .05]), W2, np.array([.1, -.2]))
    np.testing.assert_allclose(Z, [[-.91, .66], [-.93, .27]], atol=1e-12)
    np.testing.assert_allclose(H, [[0, .66], [0, .27]], atol=1e-12)
    np.testing.assert_allclose(Y, [[-.296, .1432], [-.062, -.0596]], atol=1e-12)
    print('EXERCISES PASS: analytic angles, zero vectors, input preservation, stable ties, invalid k, distances, MLP')


def check_patch_order(ImageEncoder):
    # No model needed: patchify only depends on the input array.
    encoder = ImageEncoder.__new__(ImageEncoder)
    images = np.arange(3 * 224 * 224, dtype=np.float32).reshape(1, 3, 224, 224)
    expected = np.stack([images[:, :, y:y+32, x:x+32].reshape(1, -1)
                         for y in range(0, 224, 32)
                         for x in range(0, 224, 32)], axis=1)
    # Reproduce a wrong candidate and verify the corrected implementation.
    wrong = images.reshape(1, 49, 3072)
    mismatch = int(np.count_nonzero(wrong != expected))
    assert mismatch > 0
    np.testing.assert_array_equal(encoder.patchify(images), expected)
    reversed_images = images[..., ::-1]
    reversed_expected = np.stack([reversed_images[:, :, y:y+32, x:x+32].reshape(1, -1)
                                  for y in range(0, 224, 32)
                                  for x in range(0, 224, 32)], axis=1)
    np.testing.assert_array_equal(encoder.patchify(reversed_images), reversed_expected)
    print(f'PATCH_ORDER PASS: direct-reshape bug reproduced ({mismatch} wrong elements); corrected transpose matches explicit slicing')
    print('PATCH_STRIDES PASS: non-contiguous reversed image preserves patch order')


def check_attention(MultiHeadAttention):
    rng = np.random.default_rng(47)
    batch, tokens, width, heads = 2, 4, 8, 2
    prefix = 'tiny'
    weights = {
        prefix+'.in_proj_weight': rng.normal(0, .1, (3*width, width)).astype(np.float32),
        prefix+'.in_proj_bias': rng.normal(0, .1, 3*width).astype(np.float32),
        prefix+'.out_proj.weight': rng.normal(0, .1, (width, width)).astype(np.float32),
        prefix+'.out_proj.bias': rng.normal(0, .1, width).astype(np.float32),
    }
    attention = MultiHeadAttention(weights, prefix, heads)
    x = rng.normal(size=(batch, tokens, width)).astype(np.float32)
    split = attention.split_heads(x)
    for head in range(heads):
        np.testing.assert_array_equal(split[:, head], x[:, :, head*4:(head+1)*4])
    qkv = x @ weights[prefix+'.in_proj_weight'].T + weights[prefix+'.in_proj_bias']
    # A small independent oracle keeps token/head organization explicit in loops.
    for causal in (False, True):
        joined = np.empty_like(x)
        for b in range(batch):
            for h in range(heads):
                q = qkv[b, :, h*4:(h+1)*4].astype(np.float64)
                k = qkv[b, :, width+h*4:width+(h+1)*4].astype(np.float64)
                v = qkv[b, :, 2*width+h*4:2*width+(h+1)*4].astype(np.float64)
                for t in range(tokens):
                    end = t+1 if causal else tokens
                    scores = np.array([np.dot(q[t], k[j]) / 2. for j in range(end)])
                    probabilities = np.exp(scores - scores.max())
                    probabilities /= probabilities.sum()
                    joined[b, t, h*4:(h+1)*4] = probabilities @ v[:end]
        expected = joined @ weights[prefix+'.out_proj.weight'].T + weights[prefix+'.out_proj.bias']
        np.testing.assert_allclose(attention(x, causal=causal), expected, atol=2e-7, rtol=2e-6)
    changed = x.copy()
    changed[:, 2:] += 100
    np.testing.assert_array_equal(attention(x, causal=True)[:, :2], attention(changed, causal=True)[:, :2])
    print('ATTENTION PASS: head feature order, independent scalar oracle, causal mask and future-token isolation')


def check_new_inputs(ImageEncoder, weights):
    encoder = ImageEncoder(weights)
    rng = np.random.default_rng(20261126)
    images = rng.normal(0, .5, (2, 3, 224, 224)).astype(np.float32)
    batched = encoder(images)
    separate = np.concatenate([encoder(images[:1]), encoder(images[1:])], axis=0)
    assert batched.shape == (2, 512) and batched.dtype == np.float32
    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched, separate, atol=2e-5, rtol=2e-4)
    norms = np.linalg.norm(batched, axis=-1)
    np.testing.assert_allclose(norms, 1., atol=2e-4, rtol=2e-4)
    print(f'NEW_INPUTS PASS: generated inputs B=1/B=2, batch independence max_abs={np.max(np.abs(batched-separate)):.3e}, unit norms={norms}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_dir.resolve()))
    from image_encoder import ImageEncoder, MultiHeadAttention
    from runtime import load_weights
    check_exercises()
    check_patch_order(ImageEncoder)
    check_attention(MultiHeadAttention)
    check_new_inputs(ImageEncoder, load_weights())
    print('ALL ADDITIONAL CHECKS PASSED')


if __name__ == '__main__':
    main()
