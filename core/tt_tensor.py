# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

import random

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, flat_to_multi_index


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not isinstance(cores, list) or len(cores) == 0:
            raise ValueError("cores должен быть непустым списком")
        shape: list[int] = []
        ranks: list[int] = []
        for k, core in enumerate(cores):
            if not isinstance(core, DenseTensor):
                raise TypeError("все TT-ядра должны быть экземплярами DenseTensor")
            if core.ndim != 3:
                raise ValueError("каждое TT-ядро должно быть трёхмерным тензором")
            left_rank,mode_size,right_rank =core.shape
            if k==0:
                if left_rank !=1:
                    raise ValueError("первый TT-ранг должен быть равен 1")
                ranks.append(left_rank)
            elif left_rank !=ranks[-1]:
                raise ValueError("соседние TT-ранги не совпадают")
            shape.append(mode_size)
            ranks.append(right_rank)
        if ranks[-1] !=1:
            raise ValueError("последний TT-ранг должен быть равен 1")
        self.cores = [core.copy() for core in cores]
        self.order = len(self.cores)
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)


    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        checked_shape = validate_shape(shape)
        if not isinstance(ranks, (tuple, list)):
            raise TypeError("ranks должен быть кортежем или списком")
        ranks = tuple(ranks)
        if len(ranks) == len(checked_shape) - 1:
            full_ranks = (1,) + ranks + (1,)
        elif len(ranks) == len(checked_shape) + 1:
            full_ranks = ranks
        else:
            raise ValueError("ranks должен содержать либо внутренние, либо полные TT-ранги")
        full_ranks = validate_shape(full_ranks)
        if full_ranks[0] !=1 or full_ranks[-1] !=1:
            raise ValueError("граничные TT-ранги должны быть равны 1")
        rng = random.Random(seed)
        cores: list[DenseTensor] = []
        for k, mode_size in enumerate(checked_shape):
            core_shape = (full_ranks[k], mode_size, full_ranks[k + 1])
            core_seed = rng.randrange(2**32)
            cores.append(DenseTensor.random(
                    core_shape,
                    low=-1,
                    high=1,
                    integer=False,
                    seed=core_seed,))
        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if isinstance(indices, list):
            indices = tuple(indices)
        if not isinstance(indices, tuple):
            raise TypeError("indices должен быть кортежем или списком")
        if len(indices) != self.order:
            raise IndexError("количество индексов должно совпадать с порядком TT")
        for axis, index in enumerate(indices):
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("все индексы должны быть целыми числами")
            if index < 0 or index >= self.shape[axis]:
                raise IndexError("индекс выходит за границы")
        vector = [1.0]
        for k, index in enumerate(indices):
            core = self.cores[k]
            left_rank, _, right_rank = core.shape
            next_vector = [0.0] * right_rank
            for beta in range(right_rank):
                total = 0.0
                for alpha in range(left_rank):
                    total += vector[alpha] * core[alpha, index, beta]
                next_vector[beta] = total
            vector = next_vector
        return vector[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result = DenseTensor.zeros(self.shape)
        for flat_index in range(result.size):
            index = flat_to_multi_index(flat_index, self.shape)
            result[flat_index] = self.get_element(index)
        return result

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        return compute_size(self.shape) / self.total_storage()

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        lines = [
            "TTTensor(",
            f"  order={self.order},",
            f"  shape={self.shape},",
            f"  ranks={self.ranks},",
            f"  core_sizes={self.core_sizes()},",
            f"  total_storage={self.total_storage()}",
            ")",]
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()
