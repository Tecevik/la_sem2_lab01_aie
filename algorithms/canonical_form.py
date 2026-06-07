# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores=[core.copy() for core in tt.cores]
    for k in range(tt.order-1):
        core=cores[k]
        left_rank,mode_size,right_rank=core.shape
        matrix=core.reshape((left_rank*mode_size,right_rank))
        q,s,vt=backend.svd(matrix, full_matrices=False)
        transfer=_multiply_diag_matrix(s,vt,s.shape[0],backend)
        new_rank=q.shape[1]
        cores[k]=q.reshape((left_rank,mode_size,new_rank))
        next_core = cores[k+1]
        transfer_left, transfer_right = transfer.shape
        next_left, next_mode_size, next_right = next_core.shape
        if transfer_right != next_left:
            raise ValueError("ранги transfer и следующего core не совпадают")
        updated_next = backend.zeros((transfer_left, next_mode_size, next_right))
        for alpha_new in range(transfer_left):
            for i in range(next_mode_size):
                for beta in range(next_right):
                    total=0.0
                    for alpha_old in range(transfer_right):
                        total+=(
                            transfer[alpha_new,alpha_old]
                            * next_core[alpha_old,i,beta])
                    updated_next[alpha_new,i,beta] = total
        cores[k+1] = updated_next
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]

    for k in range(tt.order -1,0,-1):
        core = cores[k]
        left_rank,mode_size,right_rank=core.shape
        matrix=core.reshape((left_rank,mode_size*right_rank))
        u,s,q=backend.svd(matrix, full_matrices=False)
        transfer=_multiply_columns_by_diag(u,s,backend)
        new_rank=q.shape[0]
        cores[k]=q.reshape((new_rank,mode_size,right_rank))
        previous_core = cores[k-1]
        previous_left,previous_mode_size,previous_right=previous_core.shape
        transfer_left,transfer_right=transfer.shape
        if previous_right !=transfer_left:
            raise ValueError("ранги предыдущего core и transfer не совпадают")
        updated_previous=backend.zeros(
            (previous_left, previous_mode_size, transfer_right))
        for alpha in range(previous_left):
            for i in range(previous_mode_size):
                for beta_new in range(transfer_right):
                    total=0.0
                    for beta_old in range(previous_right):
                        total+=(
                            previous_core[alpha, i, beta_old]
                            * transfer[beta_old, beta_new] )
                    updated_previous[alpha, i, beta_new]=total
        cores[k-1]=updated_previous
    return TTTensor(cores)


def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.ndim !=1:
        raise ValueError("S должен быть одномерным тензором")
    if S.size ==0:
        return 0
    max_singular = max(abs(value) for value in S.data)
    threshold = max(abs_tol,rel_tol *max_singular)
    rank=0
    for value in S.data:
        if abs(value)>threshold:
            rank +=1
    return rank


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
    if matrix.ndim !=2:
        raise ValueError("матрица должна быть двумерной")
    rows, cols = matrix.shape
    if rank<0 or rank>cols:
        raise ValueError("ранг выходит за допустимые пределы")
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
        raise ValueError("матрица должна быть двумерной")
    rows, cols = matrix.shape
    if rank<0 or rank>rows:
        raise ValueError("ранг выходит за допустимые пределы")
    result = backend.zeros((rank, cols))
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
    if vector.ndim !=1:
        raise ValueError("матрица должна быть двумерной")
    if rank<0 or rank>vector.size:
        raise ValueError("ранг выходит за допустимые пределы")
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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim !=1 or matrix.ndim !=2:
        raise ValueError("ожидались вектор и матрица")
    if rank<0 or rank>diag_vec.size or rank>matrix.shape[0]:
        raise ValueError("ранг выходит за допустимые пределы")
    _, cols = matrix.shape
    result = backend.zeros((rank, cols))
    for i in range(rank):
        scale = diag_vec[i]
        for j in range(cols):
            result[i,j] = scale * matrix[i,j]
    return result


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    if matrix.ndim != 2 or diag_vec.ndim != 1:
        raise ValueError("ожидались вектор и матрица")
    rows, cols = matrix.shape
    rank = diag_vec.size
    if rank > cols:
        raise ValueError("диагональный вектор слишком длинный для матрицы")
    result = backend.zeros((rows, rank))
    for i in range(rows):
        for j in range(rank):
            result[i,j] = matrix[i,j]*diag_vec[j]
    return result
