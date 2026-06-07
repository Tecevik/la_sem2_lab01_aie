# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from core.utils import check_shapes_match
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    check_shapes_match(tt1.shape,tt2.shape)
    if tt1.order==1:
        return TTTensor([backend.add(tt1.cores[0],tt2.cores[0])])
    cores: list[DenseTensor] = []
    for k in range(tt1.order):
        core1=tt1.cores[k]
        core2=tt2.cores[k]
        r1_left,mode_size,r1_right=core1.shape
        r2_left,_,r2_right=core2.shape
        if k==0:
            result=backend.zeros((1,mode_size,r1_right+r2_right))
            for i in range(mode_size):
                for beta in range(r1_right):
                    result[0,i,beta] = core1[0,i,beta]
                for beta in range(r2_right):
                    result[0,i,r1_right+beta]=core2[0,i,beta]
        elif k==tt1.order-1:
            result=backend.zeros((r1_left+r2_left,mode_size,1))
            for i in range(mode_size):
                for alpha in range(r1_left):
                    result[alpha,i,0] = core1[alpha,i,0]
                for alpha in range(r2_left):
                    result[r1_left+alpha,i,0]=core2[alpha,i,0]
        else:
            result = backend.zeros((r1_left+r2_left,mode_size,r1_right+r2_right))
            for i in range(mode_size):
                for alpha in range(r1_left):
                    for beta in range(r1_right):
                        result[alpha,i,beta] = core1[alpha,i,beta]
                for alpha in range(r2_left):
                    for beta in range(r2_right):
                        result[r1_left+alpha,i,r1_right+beta] = core2[alpha,i,beta]
        cores.append(result)
    return TTTensor(cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError("альфа должно быть числом")
    cores = [core.copy() for core in tt.cores]
    cores[0] = backend.scale(cores[0], alpha)
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    check_shapes_match(tt1.shape, tt2.shape)
    cores: list[DenseTensor] = []
    for k in range(tt1.order):
        core1=tt1.cores[k]
        core2=tt2.cores[k]
        r1_left,mode_size,r1_right=core1.shape
        r2_left,_,r2_right=core2.shape

        result=backend.zeros((r1_left*r2_left,mode_size,r1_right*r2_right))
        for i in range(mode_size):
            for a_left in range(r1_left):
                for b_left in range(r2_left):
                    left=a_left*r2_left+b_left
                    for a_right in range(r1_right):
                        for b_right in range(r2_right):
                            right=a_right*r2_right+b_right
                            result[left,i,right]=(core1[a_left,i,a_right]*core2[b_left,i,b_right])
        cores.append(result)
    return TTTensor(cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    check_shapes_match(tt1.shape,tt2.shape)

    z = [[1.0]]
    for k in range(tt1.order):
        core1=tt1.cores[k]
        core2=tt2.cores[k]
        r1_left,mode_size,r1_right = core1.shape
        r2_left,_,r2_right=core2.shape
        next_z=[[0.0 for _ in range(r2_right)] for _ in range(r1_right)]

        for i in range(mode_size):
            for a_left in range(r1_left):
                for b_left in range(r2_left):
                    z_value=z[a_left][b_left]
                    if z_value==0.0:
                        continue
                    for a_right in range(r1_right):
                        a_value = core1[a_left, i, a_right]
                        if a_value==0.0:
                            continue
                        for b_right in range(r2_right):
                            next_z[a_right][b_right] += (z_value * a_value * core2[b_left, i, b_right])
        z = next_z
    return z[0][0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    squared_norm = tt_dot(tt, tt, backend)
    return math.sqrt(max(squared_norm,0.0))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    check_shapes_match(tt1.shape, tt2.shape)
    squared_norm = (
        tt_dot(tt1, tt1, backend)
        - 2.0 * tt_dot(tt1, tt2, backend)
        + tt_dot(tt2, tt2, backend))
    return math.sqrt(max(squared_norm,0.0))
