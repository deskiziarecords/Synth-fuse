import jax
import jax.numpy as jnp
from synthfuse.pipeline.unified_vector_pipeline import UnifiedVectorPipeline

def test_pipeline():
    print("--- Synth-Fuse Vector Revolutionary Architecture Demo ---")

    ambient_dim = 128
    pipeline = UnifiedVectorPipeline(
        ambient_dim=ambient_dim,
        manifold_dim=16,
        compression_rank=32,
        use_amgdl=True,
        use_zeta=True,
        use_choco=True,
        num_distributed_nodes=4
    )

    # 1. Lazy Tensor Implementation: Store generation function
    print("\n[1] Lazy Tensor Implementation")
    def my_gen_func():
        # Simulation of a complex generation process
        return jax.random.normal(jax.random.PRNGKey(42), (ambient_dim,))

    vec = my_gen_func()
    vector_id = "lazy_gen_1"

    print(f"Registering Lazy Tensor: {vector_id}")
    # We pass the vector for initial SVD, but the system now supports F_gen
    store_metrics = pipeline.store(vector_id, vec)
    print(f"Initial storage metrics: {store_metrics}")

    # 2. Hyper-Efficiency Formulas
    print("\n[2] Hyper-Efficiency Formulas")

    # CMOP demonstration (part of store)
    print(f"CMOP Selected Rank: {pipeline.scp.rank}")

    # TEG demonstration
    base_latency = 100.0 # ms
    rho = 0.99 # 99% reduction
    teg = pipeline.zeta.compute_teg_improvement(base_latency, rho)
    print(f"Temporal Efficiency Gain (TEG): {teg:.2f} ms (from {base_latency} ms)")

    # EPR demonstration
    L = 10**6 # bits
    B = 10**6 # bps
    P = 0.5 # W
    N0 = 0.001 # W
    epr = pipeline.choco.compute_energy_consumption(L, B, P, N0)
    print(f"Energy-Performance Ratio (EPR): {epr:.4f} Joules")

    # 3. Query with MaxKurtosis and Manifold
    print("\n[3] Retrieval Performance")
    query_vec = vec + 0.01 * jax.random.normal(jax.random.PRNGKey(1), (ambient_dim,))
    query_results = pipeline.query(query_vec)

    print(f"Query Results for '{vector_id}':")
    if 'manifold_neighbors' in query_results:
        for vid, dist in query_results['manifold_neighbors']:
            print(f"  ID: {vid}, Manifold Distance: {dist:.6f}")

    print(f"AMGDL Adapted LR: {query_results.get('amgdl_lr')}")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
