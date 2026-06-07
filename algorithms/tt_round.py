# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if not isinstance(tt,TTTensor):
        raise TypeError("тт должен быть новым уменьшенным тензором")
    if max_rank is not None and max_rank<1:
        raise ValueError("ранг нужен положительный")
    if eps<0:
        raise ValueError("точность должна быть неотрицательной")
    if tt.order==1:
        return tt.copy()
    rounded = right_canonicalize(tt, backend)
    cores = [core.copy() for core in rounded.cores]
    first_norm = backend.norm(cores[0])
    if first_norm>1e-30:
        delta=eps*first_norm/math.sqrt(tt.order - 1)
    else:
        delta=0.0
    for k in range(tt.order - 1):
        core = cores[k]
        left_rank, mode_size, right_rank = core.shape
        matrix = core.reshape((left_rank * mode_size, right_rank))
        u, s, vt = backend.svd(matrix, full_matrices=False)
        rank = _compute_rank(s, delta, max_rank)
        u_trunc = _truncate_columns(u, rank, backend)
        cores[k] = u_trunc.reshape((left_rank, mode_size, rank))
        s_trunc = _truncate_vector(s, rank, backend)
        vt_trunc = _truncate_rows(vt, rank, backend)
        transfer = _multiply_diag_matrix(s_trunc,vt_trunc,rank,backend)
        next_core = cores[k+1]
        transfer_left, transfer_right = transfer.shape
        next_left, next_mode_size, next_right = next_core.shape
        if transfer_right != next_left:
            raise ValueError("ранги transfer и следующего core не совпадают")
        updated_next = backend.zeros((transfer_left, next_mode_size, next_right))
        for alpha_new in range(transfer_left):
            for i in range(next_mode_size):
                for beta in range(next_right):
                    total = 0.0
                    for alpha_old in range(transfer_right):
                        total += (
                            transfer[alpha_new, alpha_old]
                            * next_core[alpha_old, i, beta]
                        )
                    updated_next[alpha_new, i, beta] = total
        cores[k + 1] = updated_next
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    if S.ndim !=1:
        raise ValueError("s должен быть одномерным тензором")
    if S.size==0:
        return 1
    if max_rank is not None and max_rank<1:
        raise ValueError("ранг должен быть положительный")
    sigma1=abs(S[0])
    numerical_threshold = max(1e-12,1e-8*sigma1)
    numerical_rank=0
    for value in S.data:
        if abs(value)>numerical_threshold:
            numerical_rank+=1
    rank=max(1,numerical_rank)
    tail_sum=0.0
    while rank >1:
        candidate_tail=tail_sum+S[rank-1]*S[rank-1]
        if candidate_tail>delta*delta:
            break
        tail_sum=candidate_tail
        rank -=1
    if max_rank is not None:
        rank =min(rank,max_rank)
    return max(1,rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim !=2:
        raise ValueError("матрица должна быть размерности 2")
    rows,cols = matrix.shape
    if rank <0 or rank>cols:
        raise ValueError("ранг неправильный по размеру")
    result = backend.zeros((rows,rank))
    for i in range(rows):
        for j in range(rank):
            result[i,j] = matrix[i,j]
    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    if matrix.ndim !=2:
        raise ValueError("матрица должна быть разменрности 2")
    rows,cols=matrix.shape
    if rank<0 or rank >rows:
        raise ValueError("ранг неправильный по размеру")
    result=backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            result[i,j] = matrix[i,j]
    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if vector.ndim != 1:
        raise ValueError("вектор нужен размерности 1")
    if rank< 0 or rank >vector.size:
        raise ValueError("ранг неправильный по размеру")
    result = backend.zeros((rank,))
    for i in range(rank):
        result[i]=vector[i]
    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim !=1 or matrix.ndim !=2:
        raise ValueError("ожидалосб векто и матрица")
    if rank<0 or rank>diag_vec.size or rank>matrix.shape[0]:
        raise ValueError("ранг неправильный по размеру")
    _, cols = matrix.shape
    result = backend.zeros((rank,cols))
    for i in range(rank):
        scale = diag_vec[i]
        for j in range(cols):
            result[i,j] = scale * matrix[i,j]
    return result
