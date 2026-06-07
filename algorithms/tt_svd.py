# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from core.utils import compute_size
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if not isinstance(tensor, DenseTensor):
        raise TypeError("тензор должен быть DenseTensor")
    if tensor.ndim == 0:
        raise ValueError("тензор должен иметь хотя бы один режим")
    if max_rank is not None and max_rank < 1:
        raise ValueError("ранг должен быть положительным")
    if eps < 0:
        raise ValueError("точность должна быть неотрицательной")
    if tensor.ndim == 1:
        return TTTensor([tensor.reshape((1, tensor.shape[0], 1))])
    order =tensor.ndim
    original_shape =tensor.shape
    tensor_norm =backend.norm(tensor)
    if tensor_norm >1e-30:
        delta = eps * tensor_norm / math.sqrt(order - 1)
    else:
        delta = 0.0
    cores: list[DenseTensor]=[]
    previous_rank=1
    current=tensor.copy()
    for mode in range(order-1):
        mode_size=original_shape[mode]
        remaining_shape=original_shape[mode+1:]
        rows=previous_rank*mode_size
        cols=compute_size(remaining_shape)
        unfolding=current.reshape((rows, cols))
        u,s,vt=backend.svd(unfolding,full_matrices=False)
        rank=_compute_truncated_rank(s,delta,max_rank)
        u_trunc=_truncate_columns(u,rank,backend)
        cores.append(u_trunc.reshape((previous_rank,mode_size,rank)))
        s_trunc=_truncate_vector(s,rank,backend)
        vt_trunc=_truncate_rows(vt,rank,backend)
        current_matrix=_multiply_diag_matrix(s_trunc,vt_trunc,rank,backend)
        current=current_matrix.reshape((rank,)+remaining_shape)
        previous_rank=rank
    cores.append(current.reshape((previous_rank,original_shape[-1],1)))
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError("S должен быть одномерным тензором")
    if S.size == 0:
        return 1
    if max_rank is not None and max_rank < 1:
        raise ValueError("ранг должен быть положительным")
    sigma1 = abs(S[0])
    numerical_threshold = max(1e-12, 1e-8 * sigma1)
    numerical_rank=0
    for value in S.data:
        if abs(value) > numerical_threshold:
            numerical_rank += 1
    rank=max(1,numerical_rank)
    if delta>0.0:
        tail_sum=0.0
        while rank > 1:
            candidate_tail = tail_sum + S[rank - 1] * S[rank - 1]
            if candidate_tail > delta * delta:
                break
            tail_sum = candidate_tail
            rank -= 1
    if max_rank is not None:
        rank = min(rank, max_rank)
    return max(1, rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("матрица должна быть двумерной")
    rows, cols = matrix.shape
    if rank < 0 or rank > cols:
        raise ValueError("ранг неправильный по размеру")
    result = backend.zeros((rows, rank))
    for i in range(rows):
        for j in range(rank):
            result[i, j] = matrix[i, j]
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
        raise ValueError("матрица должна быть двумерной")
    rows, cols = matrix.shape
    if rank<0 or rank>rows:
        raise ValueError("ранг неправильный по размеру")
    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            result[i, j] = matrix[i, j]
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
        raise ValueError("матрица должна быть одномерной")
    if rank < 0 or rank >vector.size:
        raise ValueError("ранг неправильный по размеру")
    result = backend.zeros((rank,))
    for i in range(rank):
        result[i] = vector[i]
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
    if diag_vec.ndim != 1 or matrix.ndim != 2:
        raise ValueError("ожидалась матрица и вектор")
    if rank < 0 or rank > diag_vec.size or rank > matrix.shape[0]:
        raise ValueError("ранг неправильный по размеру")
    _, cols = matrix.shape
    result = backend.zeros((rank, cols))
    for i in range(rank):
        scale = diag_vec[i]
        for j in range(cols):
            result[i, j] = scale * matrix[i, j]
    return result
