from typing import NamedTuple

import equinox as eqx
import jax
import optax
from jaxtyping import PyTree


class TrainState[TModel: eqx.Module](NamedTuple):
    """Container for model and optimizer state."""

    model: TModel
    opt_state: optax.OptState

    @classmethod
    def create[T: eqx.Module](
        cls: type, model: T, optim: optax.GradientTransformation
    ) -> "TrainState[T]":
        return cls(
            model=model,
            opt_state=optim.init(eqx.filter(model, eqx.is_array)),
        )

    def apply_gradients(
        self, optim: optax.GradientTransformation, grads: PyTree
    ) -> "TrainState":
        updates, new_opt_state = optim.update(
            grads, self.opt_state, eqx.filter(self.model, eqx.is_array)
        )
        new_model = eqx.apply_updates(self.model, updates)
        return self._replace(model=new_model, opt_state=new_opt_state)


def ensemble_to_list(models: eqx.Module, num_models: int) -> list[eqx.Module]:
    """Convert an Equinox Module ensemble to a list of Modules."""

    ensemble_params, static = eqx.partition(models, eqx.is_array)
    model_params = [
        jax.tree.map(lambda x: x[i], ensemble_params) for i in range(num_models)
    ]
    return [eqx.combine(model_param, static) for model_param in model_params]
