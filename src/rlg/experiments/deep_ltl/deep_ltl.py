"""
Functions for reproducing the DeepLTL experiments in the paper.
"""

import logging
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
from jax.experimental import io_callback
from jaxltl.deep_ltl.eval.eval import Evaluator
from jaxltl.deep_ltl.eval.utils import (
    build_env,
    load_batched_models,
    preprocess_formulas,
)
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.ltl.automata.jax_ldba import JaxLDBA
from omegaconf import DictConfig
from tqdm import tqdm

from rlg import MODEL_DIR
from rlg.experiments.deep_ltl.sampler import ReachAvoidFormulaSampler

logger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> pd.DataFrame:
    # build environment
    env, env_params = build_env(cfg, None)

    # sample formulas
    logger.info("Sampling formulas...")
    sampler = ReachAvoidFormulaSampler(
        depth=tuple(cfg.tasks.depth),
        reach=tuple(cfg.tasks.reach),
        avoid=tuple(cfg.tasks.avoid),
        propositions=env.propositions,
    )
    formulas = [sampler.sample() for _ in range(cfg.num_tasks)]

    # construct ldbas and batched sequences for all formulas
    logger.info("Building LDBAs and batched sequences...")
    ldba, batched_seqs = preprocess_formulas(formulas, env)

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    path = MODEL_DIR / "deep_ltl.eqx"
    models = load_batched_models(cfg, path, env, env_params, key=model_key)
    params, static = eqx.partition(models, eqx.is_array)
    params = jax.tree.map(lambda x: x[0], params)  # use first seed
    model = eqx.combine(params, static)

    # set up evaluator
    # TODO: run evaluator with smaller number of episodes for speed?
    evaluator = Evaluator(num_episodes=cfg.num_episodes, discount=0.99)
    eval_fn = jax.vmap(evaluator.eval, in_axes=(None, None, None, None, 0, 0, None))

    # evaluate
    key, eval_key = jax.random.split(key)
    logger.info("Starting evaluation...")
    start = time.time()
    assert cfg.num_tasks % cfg.batch_size == 0, (
        "num_tasks must be divisible by batch_size"
    )
    num_batches = cfg.num_tasks // cfg.batch_size
    batches = jax.tree.map(
        lambda x: x.reshape((num_batches, cfg.batch_size) + x.shape[1:]),
        (ldba, batched_seqs),
    )
    pbar = tqdm(total=num_batches, desc="Evaluating batches", leave=False)

    def update_progress():
        pbar.update(1)

    def batch_iter(key: jax.Array, batch: tuple[JaxLDBA, JaxReachAvoidSequence]):
        key, eval_key = jax.random.split(key)
        ldba_batch, batched_seqs_batch = batch
        metrics, returns, lengths, trajs = eval_fn(
            model,
            cfg.deterministic,
            env,  # type: ignore
            env_params,
            ldba_batch,
            batched_seqs_batch,
            eval_key,
        )  # shape: (batch_size, num_episodes)
        io_callback(update_progress, None)
        return key, metrics

    _, per_batch_metrics = eqx.filter_jit(jax.lax.scan)(batch_iter, eval_key, batches)
    metrics = jnp.concatenate(per_batch_metrics, axis=0)
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # process results
    return process_results(formulas, metrics)


def process_results(formulas: list[str], metrics: jax.Array) -> pd.DataFrame:
    """Process evaluation results into a DataFrame."""
    num_episodes = metrics.shape[1]
    rows = []
    for i, formula in enumerate(formulas):
        metrics_i = metrics[i]  # (num_episodes,)

        rows.append(
            {
                "task_id": i,
                "formula": formula,
                "num_episodes": int(num_episodes),
                "num_successes": float(jnp.sum(metrics_i)),
                "min": float(jnp.min(metrics_i)),
                "max": float(jnp.max(metrics_i)),
            }
        )

    return pd.DataFrame(rows)
