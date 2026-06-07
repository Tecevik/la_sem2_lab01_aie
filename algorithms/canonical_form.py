# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
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
        q,transfer=backend.qr(matrix)
        new_rank=q.shape[1]
        cores[k]=q.reshape((left_rank,mode_size,new_rank))
        next_core = cores[k+1]
        next_left, next_mode_size, next_right = next_core.shape
        if transfer.shape[1] != next_left:
            raise ValueError("ранги transfer и следующего core не совпадают")
        next_matrix = next_core.reshape((next_left,next_mode_size*next_right))
        updated_next = backend.matmul(transfer,next_matrix)
        cores[k+1] = updated_next.reshape((new_rank,next_mode_size,next_right))
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
        if left_rank <= mode_size*right_rank:
            q_transposed,r_transposed=backend.qr(backend.transpose(matrix))
            q=backend.transpose(q_transposed)
            transfer=backend.transpose(r_transposed)
        else:
            u,s,q=backend.svd(matrix, full_matrices=False)
            transfer=backend.matmul(u, backend.diag(s))
        new_rank=q.shape[0]
        cores[k]=q.reshape((new_rank,mode_size,right_rank))
        previous_core = cores[k-1]
        previous_left,previous_mode_size,previous_right=previous_core.shape
        if previous_right !=transfer.shape[0]:
            raise ValueError("ранги предыдущего core и transfer не совпадают")
        previous_matrix=previous_core.reshape(
            (previous_left*previous_mode_size,previous_right))
        updated_previous=backend.matmul(previous_matrix,transfer)
        cores[k-1]=updated_previous.reshape(
            (previous_left,previous_mode_size,new_rank))
    return TTTensor(cores)
