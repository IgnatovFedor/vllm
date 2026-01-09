import torch
import pytest
import numpy as np
from scipy import stats

from vllm.poc.gpu_random import (
    generate_inputs,
    generate_target,
    generate_householder_vector,
    apply_householder,
    random_pick_indices,
    generate_haar_orthogonal_matrices,
    _uniform,
    _normal,
    _seed_from_string,
)

BLOCK_HASH = "test_block_hash_12345"
PUBLIC_KEY = "test_public_key"


# === Input Generation Tests ===

def test_inputs_determinism():
    device = torch.device("cuda:0")
    nonces = [1, 2, 3]

    inputs1 = generate_inputs(BLOCK_HASH, PUBLIC_KEY, nonces, dim=128, seq_len=16, device=device)
    inputs2 = generate_inputs(BLOCK_HASH, PUBLIC_KEY, nonces, dim=128, seq_len=16, device=device)

    assert torch.allclose(inputs1, inputs2)


def test_inputs_different_nonces():
    device = torch.device("cuda:0")

    inputs1 = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [1], dim=128, seq_len=16, device=device)
    inputs2 = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [2], dim=128, seq_len=16, device=device)

    assert not torch.allclose(inputs1, inputs2)


def test_different_block_hash_produces_different_inputs():
    """Different block_hash produces different input tensors"""
    device = torch.device("cuda:0")
    inputs1 = generate_inputs("hash1", PUBLIC_KEY, [0], dim=128, seq_len=16, device=device)
    inputs2 = generate_inputs("hash2", PUBLIC_KEY, [0], dim=128, seq_len=16, device=device)
    assert not torch.allclose(inputs1, inputs2)


def test_different_public_key_produces_different_inputs():
    """Different public_key produces different input tensors"""
    device = torch.device("cuda:0")
    inputs1 = generate_inputs(BLOCK_HASH, "node1", [0], dim=128, seq_len=16, device=device)
    inputs2 = generate_inputs(BLOCK_HASH, "node2", [0], dim=128, seq_len=16, device=device)
    assert not torch.allclose(inputs1, inputs2)


def test_cpu_gpu_inputs_match():
    """CPU and GPU produce identical inputs (cross-device reproducibility)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    inputs_cpu = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=128, seq_len=16, device=cpu)
    inputs_gpu = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=128, seq_len=16, device=gpu)

    # Allow small tolerance for float32->float16 conversion differences between CPU/GPU
    assert torch.allclose(inputs_cpu, inputs_gpu.cpu(), rtol=1e-3, atol=1e-3)


# === Target Generation Tests ===

def test_target_unit_vector():
    device = torch.device("cuda:0")
    target = generate_target(BLOCK_HASH, PUBLIC_KEY, dim=1000, device=device)

    assert abs(target.norm().item() - 1.0) < 1e-5


def test_different_block_hash():
    device = torch.device("cuda:0")

    target1 = generate_target("hash1", PUBLIC_KEY, dim=1000, device=device)
    target2 = generate_target("hash2", PUBLIC_KEY, dim=1000, device=device)

    assert not torch.allclose(target1, target2)


def test_cpu_gpu_target_match():
    """CPU and GPU produce identical target vectors (cross-device reproducibility)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    target_cpu = generate_target(BLOCK_HASH, PUBLIC_KEY, dim=1000, device=cpu)
    target_gpu = generate_target(BLOCK_HASH, PUBLIC_KEY, dim=1000, device=gpu)

    assert torch.allclose(target_cpu, target_gpu.cpu())


# === Householder Transform Tests ===

def test_householder_vector_is_unit():
    """Householder reflection vector should be unit vector"""
    device = torch.device("cuda:0")
    v = generate_householder_vector("test_seed", dim=1024, device=device)
    assert abs(v.norm().item() - 1.0) < 1e-5


def test_householder_vector_determinism():
    """Same seed produces same Householder vector"""
    device = torch.device("cuda:0")
    v1 = generate_householder_vector("test_seed", dim=1024, device=device)
    v2 = generate_householder_vector("test_seed", dim=1024, device=device)
    assert torch.allclose(v1, v2)


def test_householder_vector_different_seeds():
    """Different seeds produce different Householder vectors"""
    device = torch.device("cuda:0")
    v1 = generate_householder_vector("seed1", dim=1024, device=device)
    v2 = generate_householder_vector("seed2", dim=1024, device=device)
    assert not torch.allclose(v1, v2)


def test_apply_householder_preserves_norm():
    """Householder reflection preserves vector norm (orthogonal transform)"""
    device = torch.device("cuda:0")
    x = torch.randn(10, 1024, device=device)
    v = generate_householder_vector("test", dim=1024, device=device)
    
    x_transformed = apply_householder(x, v)
    
    # Norms should be preserved
    norms_before = x.norm(dim=1)
    norms_after = x_transformed.norm(dim=1)
    assert torch.allclose(norms_before, norms_after, rtol=1e-4)


def test_apply_householder_is_involutory():
    """Householder reflection applied twice returns original (H @ H @ x = x)"""
    device = torch.device("cuda:0")
    x = torch.randn(5, 1024, device=device)
    v = generate_householder_vector("test", dim=1024, device=device)
    
    x_once = apply_householder(x, v)
    x_twice = apply_householder(x_once, v)
    
    # Numerical precision can accumulate, so use larger tolerance
    assert torch.allclose(x, x_twice, rtol=1e-3, atol=1e-5)


def test_cpu_gpu_householder_match():
    """CPU and GPU produce identical Householder vectors"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    
    v_cpu = generate_householder_vector("test", dim=1024, device=cpu)
    v_gpu = generate_householder_vector("test", dim=1024, device=gpu)
    
    assert torch.allclose(v_cpu, v_gpu.cpu())


# === Random Pick Indices Tests ===

def test_random_pick_indices_determinism():
    """Same inputs produce same indices"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2]
    
    idx1 = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, nonces, dim=1000, k=64, device=device)
    idx2 = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, nonces, dim=1000, k=64, device=device)
    
    assert torch.equal(idx1, idx2)


def test_random_pick_indices_shape():
    """random_pick_indices returns correct shape"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2, 3, 4]
    k = 64
    
    indices = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, nonces, dim=1000, k=k, device=device)
    
    assert indices.shape == (len(nonces), k)
    assert indices.dtype == torch.int64


def test_random_pick_indices_range():
    """Indices are within valid range [0, dim)"""
    device = torch.device("cuda:0")
    dim = 1000
    k = 64
    
    indices = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=dim, k=k, device=device)
    
    assert (indices >= 0).all()
    assert (indices < dim).all()


def test_random_pick_indices_uniqueness():
    """Each nonce's indices are unique (no replacement)"""
    device = torch.device("cuda:0")
    k = 64
    
    indices = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0], dim=1000, k=k, device=device)
    
    # Check all indices are unique for this nonce
    unique_count = len(torch.unique(indices[0]))
    assert unique_count == k


def test_random_pick_indices_different_nonces():
    """Different nonces produce different indices"""
    device = torch.device("cuda:0")
    
    idx1 = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0], dim=1000, k=64, device=device)
    idx2 = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [1], dim=1000, k=64, device=device)
    
    # Different nonces should produce different index sets (with high probability)
    assert not torch.equal(idx1, idx2)


def test_random_pick_indices_invalid_k():
    """Invalid k values raise ValueError"""
    device = torch.device("cuda:0")
    
    with pytest.raises(ValueError):
        random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0], dim=1000, k=0, device=device)
    
    with pytest.raises(ValueError):
        random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0], dim=1000, k=1001, device=device)


def test_cpu_gpu_random_pick_indices_match():
    """CPU and GPU produce same set of indices (order may differ due to topk impl)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    
    idx_cpu = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=1000, k=64, device=cpu)
    idx_gpu = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=1000, k=64, device=gpu)
    
    # Compare sorted indices since topk order is implementation-defined
    for i in range(idx_cpu.shape[0]):
        cpu_sorted = torch.sort(idx_cpu[i]).values
        gpu_sorted = torch.sort(idx_gpu[i].cpu()).values
        assert torch.equal(cpu_sorted, gpu_sorted)


# === Haar Orthogonal Matrices Tests ===

def test_haar_orthogonal_matrices_determinism():
    """Same inputs produce same matrices"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2]
    
    Q1 = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, nonces, k=64, device=device)
    Q2 = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, nonces, k=64, device=device)
    
    assert torch.allclose(Q1, Q2)


def test_haar_orthogonal_matrices_shape():
    """generate_haar_orthogonal_matrices returns correct shape"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2, 3, 4]
    k = 64
    
    Q = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, nonces, k=k, device=device)
    
    assert Q.shape == (len(nonces), k, k)


def test_haar_orthogonal_matrices_orthogonality():
    """Generated matrices are orthogonal (Q @ Q^T = I)"""
    device = torch.device("cuda:0")
    k = 64
    
    Q = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], k=k, device=device)
    
    for i in range(Q.shape[0]):
        QQt = Q[i] @ Q[i].T
        identity = torch.eye(k, device=device, dtype=Q.dtype)
        assert torch.allclose(QQt, identity, atol=1e-5)


def test_haar_orthogonal_matrices_determinant():
    """Orthogonal matrices have determinant +/- 1"""
    device = torch.device("cuda:0")
    k = 32  # Smaller k for faster determinant computation
    
    Q = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], k=k, device=device)
    
    for i in range(Q.shape[0]):
        det = torch.linalg.det(Q[i])
        assert abs(abs(det.item()) - 1.0) < 1e-4


def test_haar_orthogonal_matrices_different_nonces():
    """Different nonces produce different matrices"""
    device = torch.device("cuda:0")
    
    Q1 = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0], k=64, device=device)
    Q2 = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [1], k=64, device=device)
    
    assert not torch.allclose(Q1, Q2)


def test_haar_orthogonal_matrices_invalid_k():
    """Invalid k values raise ValueError"""
    device = torch.device("cuda:0")
    
    with pytest.raises(ValueError):
        generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0], k=0, device=device)


def test_cpu_gpu_haar_matrices_match():
    """CPU and GPU produce identical Haar matrices (small numerical tolerance)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    
    Q_cpu = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0, 1], k=32, device=cpu)
    Q_gpu = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0, 1], k=32, device=gpu)
    
    # QR decomposition may have small numerical differences across devices
    assert torch.allclose(Q_cpu, Q_gpu.cpu(), rtol=1e-4, atol=1e-5)


# =============================================================================
# DISTRIBUTION TESTS
# =============================================================================
# These tests verify that random functions produce correct statistical 
# distributions, not just deterministic behavior.

# === _uniform() Distribution Tests ===

def test_uniform_distribution_range():
    """_uniform() produces values strictly in [0, 1)"""
    device = torch.device("cuda:0")
    n = 100000
    
    for seed in [42, 123, 999]:
        u = _uniform(seed, n, device)
        assert (u >= 0.0).all(), "Values below 0 found"
        assert (u < 1.0).all(), "Values >= 1 found"


def test_uniform_distribution_mean_variance():
    """_uniform() has correct mean (0.5) and variance (1/12)"""
    device = torch.device("cuda:0")
    n = 100000
    
    # Collect samples from multiple seeds for robustness
    all_samples = []
    for seed in range(100):
        u = _uniform(seed, n, device)
        all_samples.append(u)
    
    combined = torch.cat(all_samples)
    mean = combined.mean().item()
    var = combined.var().item()
    
    # Expected: mean=0.5, var=1/12≈0.0833
    # With 10M samples, standard error of mean ≈ sqrt(var/n) ≈ 0.00009
    # Use 5-sigma tolerance for robustness
    expected_mean = 0.5
    expected_var = 1.0 / 12.0
    
    assert abs(mean - expected_mean) < 0.005, f"Mean {mean} too far from {expected_mean}"
    assert abs(var - expected_var) < 0.005, f"Variance {var} too far from {expected_var}"


def test_uniform_distribution_ks_test():
    """_uniform() passes Kolmogorov-Smirnov test against uniform distribution"""
    device = torch.device("cuda:0")
    n = 10000
    
    # Test multiple seeds
    passing_seeds = 0
    for seed in range(20):
        u = _uniform(seed, n, device).cpu().numpy()
        
        # KS test against uniform(0, 1)
        stat, p_value = stats.kstest(u, 'uniform', args=(0, 1))
        
        # At alpha=0.01, we expect ~99% to pass
        if p_value > 0.01:
            passing_seeds += 1
    
    # At least 17 out of 20 should pass (allowing for statistical variation)
    assert passing_seeds >= 17, f"Only {passing_seeds}/20 seeds passed KS test"


# === _normal() Distribution Tests ===

def test_normal_distribution_mean_std():
    """_normal() has correct mean (0) and std (1)"""
    device = torch.device("cuda:0")
    n = 100000
    
    # Collect samples from multiple seeds
    all_samples = []
    for seed in range(100):
        z = _normal(seed, n, device)
        all_samples.append(z)
    
    combined = torch.cat(all_samples)
    mean = combined.mean().item()
    std = combined.std().item()
    
    # With 10M samples, tolerances should be tight
    assert abs(mean) < 0.01, f"Mean {mean} too far from 0"
    assert abs(std - 1.0) < 0.01, f"Std {std} too far from 1.0"


def test_normal_distribution_skewness_kurtosis():
    """_normal() has correct skewness (~0) and kurtosis (~3)"""
    device = torch.device("cuda:0")
    n = 100000
    
    # Collect large sample
    all_samples = []
    for seed in range(50):
        z = _normal(seed, n, device)
        all_samples.append(z)
    
    combined = torch.cat(all_samples).cpu().numpy()
    
    skewness = stats.skew(combined)
    kurtosis = stats.kurtosis(combined, fisher=False)  # Pearson kurtosis (normal=3)
    
    # Normal distribution: skewness=0, kurtosis=3
    assert abs(skewness) < 0.05, f"Skewness {skewness} too far from 0"
    assert abs(kurtosis - 3.0) < 0.1, f"Kurtosis {kurtosis} too far from 3"


def test_normal_distribution_ks_test():
    """_normal() passes Kolmogorov-Smirnov test against normal distribution"""
    device = torch.device("cuda:0")
    n = 10000
    
    passing_seeds = 0
    for seed in range(20):
        z = _normal(seed, n, device).cpu().numpy()
        
        # KS test against N(0, 1)
        stat, p_value = stats.kstest(z, 'norm', args=(0, 1))
        
        if p_value > 0.01:
            passing_seeds += 1
    
    # At least 17 out of 20 should pass
    assert passing_seeds >= 17, f"Only {passing_seeds}/20 seeds passed KS test"


# === generate_inputs() Distribution Tests ===

def test_generate_inputs_gaussian_distribution():
    """generate_inputs() produces Gaussian-distributed elements (mean=0, std≈1)"""
    device = torch.device("cuda:0")
    dim = 512
    seq_len = 32
    
    # Collect samples from many nonces
    all_samples = []
    for batch_start in range(0, 1000, 100):
        nonces = list(range(batch_start, batch_start + 100))
        inputs = generate_inputs(BLOCK_HASH, PUBLIC_KEY, nonces, dim=dim, seq_len=seq_len, device=device)
        all_samples.append(inputs.flatten().float())
    
    combined = torch.cat(all_samples)
    mean = combined.mean().item()
    std = combined.std().item()
    
    # Elements should be N(0, 1)
    assert abs(mean) < 0.05, f"Mean {mean} too far from 0"
    assert abs(std - 1.0) < 0.1, f"Std {std} too far from 1.0"


def test_generate_inputs_per_nonce_independence():
    """Different nonces produce statistically independent inputs"""
    device = torch.device("cuda:0")
    dim = 256
    seq_len = 16
    
    # Generate inputs for two consecutive nonces
    inputs = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [0, 1], dim=dim, seq_len=seq_len, device=device)
    
    # Flatten each nonce's input
    v0 = inputs[0].flatten().float()
    v1 = inputs[1].flatten().float()
    
    # Compute correlation coefficient
    v0_centered = v0 - v0.mean()
    v1_centered = v1 - v1.mean()
    correlation = (v0_centered * v1_centered).mean() / (v0.std() * v1.std())
    
    # Correlation should be near zero (independent samples)
    assert abs(correlation.item()) < 0.1, f"Correlation {correlation.item()} too high for independent samples"


# === generate_target() Distribution Tests ===

def test_generate_target_uniform_on_sphere():
    """generate_target() produces vectors uniformly distributed on unit sphere"""
    device = torch.device("cuda:0")
    dim = 100
    n_samples = 5000
    
    # Collect many targets with different seeds
    targets = []
    for i in range(n_samples):
        target = generate_target(f"block_{i}", f"key_{i}", dim=dim, device=device)
        targets.append(target)
    
    targets = torch.stack(targets)  # [n_samples, dim]
    
    # Test 1: Each component should have mean 0
    component_means = targets.mean(dim=0)
    max_mean = component_means.abs().max().item()
    assert max_mean < 0.1, f"Max component mean {max_mean} too far from 0"
    
    # Test 2: Each component should have variance 1/dim
    component_vars = targets.var(dim=0)
    expected_var = 1.0 / dim
    mean_var = component_vars.mean().item()
    assert abs(mean_var - expected_var) < 0.01, f"Mean variance {mean_var} too far from {expected_var}"
    
    # Test 3: Components should be uncorrelated
    # Sample correlation matrix (should be near identity * 1/dim)
    targets_centered = targets - targets.mean(dim=0, keepdim=True)
    cov = (targets_centered.T @ targets_centered) / n_samples
    
    # Off-diagonal elements should be near zero
    identity = torch.eye(dim, device=device, dtype=cov.dtype)
    off_diag = cov * (1 - identity)
    max_off_diag = off_diag.abs().max().item()
    assert max_off_diag < 0.05, f"Max off-diagonal covariance {max_off_diag} too high"


def test_generate_target_component_distribution():
    """Individual components of targets follow correct marginal distribution"""
    device = torch.device("cuda:0")
    dim = 64
    n_samples = 10000
    
    # Collect first component from many targets
    first_components = []
    for i in range(n_samples):
        target = generate_target(f"block_{i}", f"key_{i}", dim=dim, device=device)
        first_components.append(target[0].item())
    
    first_components = np.array(first_components)
    
    # For uniform on sphere in dim D, each component has mean 0 and variance 1/D
    mean = first_components.mean()
    var = first_components.var()
    
    expected_var = 1.0 / dim
    assert abs(mean) < 0.05, f"Component mean {mean} too far from 0"
    assert abs(var - expected_var) < 0.01, f"Component variance {var} too far from {expected_var}"


# === generate_householder_vector() Distribution Tests ===

def test_householder_vector_uniform_on_sphere():
    """generate_householder_vector() produces vectors uniformly distributed on sphere"""
    device = torch.device("cuda:0")
    dim = 100
    n_samples = 5000
    
    # Collect many Householder vectors with different seeds
    vectors = []
    for i in range(n_samples):
        v = generate_householder_vector(f"seed_{i}", dim=dim, device=device)
        vectors.append(v)
    
    vectors = torch.stack(vectors)  # [n_samples, dim]
    
    # Test 1: Each component should have mean 0
    component_means = vectors.mean(dim=0)
    max_mean = component_means.abs().max().item()
    assert max_mean < 0.1, f"Max component mean {max_mean} too far from 0"
    
    # Test 2: Each component should have variance 1/dim
    component_vars = vectors.var(dim=0)
    expected_var = 1.0 / dim
    mean_var = component_vars.mean().item()
    assert abs(mean_var - expected_var) < 0.01, f"Mean variance {mean_var} too far from {expected_var}"


# === random_pick_indices() Distribution Tests ===

def test_random_pick_indices_uniform_selection():
    """Each dimension has equal probability of being selected"""
    device = torch.device("cuda:0")
    dim = 100
    k = 10
    n_nonces = 10000
    
    # Count how often each index is selected
    counts = torch.zeros(dim, device=device, dtype=torch.int64)
    
    for batch_start in range(0, n_nonces, 1000):
        nonces = list(range(batch_start, min(batch_start + 1000, n_nonces)))
        indices = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, nonces, dim=dim, k=k, device=device)
        
        # Flatten and count
        for idx in indices.flatten():
            counts[idx] += 1
    
    # Expected count per index: n_nonces * k / dim
    expected_count = n_nonces * k / dim
    counts_float = counts.float()
    
    # Chi-square test: sum of (observed - expected)^2 / expected
    chi2 = ((counts_float - expected_count) ** 2 / expected_count).sum().item()
    
    # Chi-square with dim-1 degrees of freedom
    # At alpha=0.01, critical value for df=99 is ~134.6
    critical_value = stats.chi2.ppf(0.99, df=dim - 1)
    
    assert chi2 < critical_value, f"Chi-square {chi2} exceeds critical value {critical_value}"


def test_random_pick_indices_coverage():
    """Over many nonces, all dimensions should be selected with roughly equal frequency"""
    device = torch.device("cuda:0")
    dim = 200
    k = 20
    n_nonces = 5000
    
    # Count selections
    counts = torch.zeros(dim, device=device, dtype=torch.int64)
    
    nonces = list(range(n_nonces))
    for batch_start in range(0, n_nonces, 500):
        batch_nonces = nonces[batch_start:batch_start + 500]
        indices = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, batch_nonces, dim=dim, k=k, device=device)
        for idx in indices.flatten():
            counts[idx] += 1
    
    counts_float = counts.float()
    
    # All dimensions should be selected at least once
    assert (counts > 0).all(), "Some dimensions were never selected"
    
    # Ratio of max to min count should not be too extreme
    # Expected: each dim selected ~n_nonces * k / dim = 5000 * 20 / 200 = 500 times
    # With Poisson-like distribution, ratio should be reasonable
    ratio = counts.max().item() / max(counts.min().item(), 1)
    assert ratio < 3.0, f"Selection ratio {ratio} too extreme (suggests non-uniform distribution)"


# === generate_haar_orthogonal_matrices() Distribution Tests ===

def test_haar_first_column_uniform_on_sphere():
    """First column of Haar matrices is uniformly distributed on sphere"""
    device = torch.device("cuda:0")
    k = 32
    n_samples = 3000
    
    # Collect first columns
    first_cols = []
    for i in range(n_samples):
        Q = generate_haar_orthogonal_matrices(f"block_{i}", f"key_{i}", [0], k=k, device=device)
        first_cols.append(Q[0, :, 0])  # First column of the single matrix
    
    first_cols = torch.stack(first_cols)  # [n_samples, k]
    
    # Test 1: Component means should be 0
    component_means = first_cols.mean(dim=0)
    max_mean = component_means.abs().max().item()
    assert max_mean < 0.1, f"Max component mean {max_mean} too far from 0"
    
    # Test 2: Component variances should be 1/k
    component_vars = first_cols.var(dim=0)
    expected_var = 1.0 / k
    mean_var = component_vars.mean().item()
    assert abs(mean_var - expected_var) < 0.02, f"Mean variance {mean_var} too far from {expected_var}"


def test_haar_determinant_distribution():
    """Haar matrices have determinant +1 and -1 with roughly equal probability"""
    device = torch.device("cuda:0")
    k = 16  # Small k for fast determinant computation
    n_samples = 1000
    
    pos_count = 0
    neg_count = 0
    
    for i in range(n_samples):
        Q = generate_haar_orthogonal_matrices(f"block_{i}", f"key_{i}", [0], k=k, device=device)
        det = torch.linalg.det(Q[0])
        
        if det > 0:
            pos_count += 1
        else:
            neg_count += 1
    
    # Should be roughly 50/50
    ratio = pos_count / n_samples
    
    # Allow deviation from 0.5 - use binomial confidence interval
    # For n=1000, p=0.5, 99% CI is roughly [0.46, 0.54]
    assert 0.40 < ratio < 0.60, f"Determinant ratio {ratio} too far from 0.5"


def test_haar_all_columns_uniform():
    """All columns of Haar matrices are uniformly distributed on sphere"""
    device = torch.device("cuda:0")
    k = 16
    n_samples = 2000
    
    # Test each column
    for col_idx in range(k):
        cols = []
        for i in range(n_samples):
            Q = generate_haar_orthogonal_matrices(f"block_{i}", f"key_{i}", [0], k=k, device=device)
            cols.append(Q[0, :, col_idx])
        
        cols = torch.stack(cols)
        
        # Component means should be 0
        component_means = cols.mean(dim=0)
        max_mean = component_means.abs().max().item()
        assert max_mean < 0.15, f"Column {col_idx}: max mean {max_mean} too far from 0"
        
        # Component variances should be 1/k
        component_vars = cols.var(dim=0)
        expected_var = 1.0 / k
        mean_var = component_vars.mean().item()
        assert abs(mean_var - expected_var) < 0.03, f"Column {col_idx}: mean var {mean_var} too far from {expected_var}"
