from dataclasses import dataclass

import jax.numpy as np
from jax import grad, jit, random, vmap
from jax.tree_util import register_pytree_node
from collections import namedtuple


def differentiable(cls):

    cls.__differentiable__ = True

    if hasattr(cls, "__dataclass_fields__"):
        fields = {name: value.type for name, value in cls.__dataclass_fields__.items()}
    elif hasattr(cls, "__annotations__"):
        fields = cls.__annotations__.copy()
    else:
        raise ValueError(f"Invalid type: {cls}")

    diff_fields = [
        name
        for name, type_ in fields.items()
        if type_ == differentiable or getattr(type_, "__differentiable__", False)
    ]

    cls.Tangent = namedtuple(f"{cls.__name__}Tangent", diff_fields)

    def flatten(self):
        diff_vars = [getattr(self, name) for name in diff_fields]

        return diff_vars, None

    def unflatten(_self, diff_vars):
        return cls.Tangent(*diff_vars)

    def point(self):
        return cls.Tangent(*[getattr(self, name) for name in diff_fields])

    cls.point = point

    register_pytree_node(cls, flatten, unflatten)

    def move(self, tangent):

        point = self.point()
        next_point = []

        for point_value, tangent_value in zip(point, tangent):

            if hasattr(point_value, "__differentiable__"):
                next_value = point_value.move(tangent_value)
            else:
                next_value = point_value + tangent_value

            next_point.append(next_value)

        return cls.Tangent(*next_point)

    cls.move = move

    def update(self, next_point):

        for field_name in diff_fields:
            value = getattr(next_point, field_name)
            setattr(self, field_name, value)

    cls.update = update

    return cls


@differentiable
@dataclass
class Model:
    a: int
    b: differentiable
    c: differentiable
    d: str = "value"

    # def __init__(self, a, b, c, d="value2"):
    #     self.a = a
    #     self.b = b
    #     self.c = c
    #     self.d = d
    #     self.f = 1


def loss(s):
    return 3 * np.sum(s.b) + 5 * np.sum(s.c)


dlosss = grad(loss)


model = Model(a=1, b=np.ones((1,)), c=np.ones((2,)))

dmodel = dlosss(model)

print(model)
print(dmodel)

next_point = model.move(dmodel)
model.update(next_point)

print(next_point)
print(model)
