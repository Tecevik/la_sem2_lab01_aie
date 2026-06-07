# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
    check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)
        if data is None:
            self.data = [fill] * self.size
        else:
            if len(data) != self.size:
                raise ValueError(f"длина данных {len(data)} не соответствует размеру тензора {self.size}")
            self.data = list(data)

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        checked_shape = validate_shape(shape)
        size = compute_size(checked_shape)
        rng = random.Random(seed)
        if integer:
            data = [rng.randint(low, high) for _ in range(size)]
        else:
            data = [rng.uniform(low, high) for _ in range(size)]
        return DenseTensor(checked_shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        if not isinstance(nested, list):
            raise TypeError("nested должен быть списком")
        def infer_shape(obj) -> tuple[int, ...]:
            if not isinstance(obj, list):
                return ()
            if len(obj) == 0:
                raise ValueError("nested должен быть сне ппустым")
            first_shape = infer_shape(obj[0])
            for item in obj[1:]:
                if infer_shape(item) != first_shape:
                    raise ValueError("вложенный список должен быть прямоугольным")
            return (len(obj),) + first_shape
        def flatten(obj, result: list[float]) -> None:
            if isinstance(obj, list):
                for item in obj:
                    flatten(item, result)
            else:
                result.append(obj)
        shape = infer_shape(nested)
        data: list[float] = []
        flatten(nested, data)
        return DenseTensor(shape, data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int) and not isinstance(multi_index, bool):
            return flat_to_multi_index(multi_index, self.shape)

        if isinstance(multi_index, list):
            multi_index = tuple(multi_index)
        if not isinstance(multi_index, tuple):
            raise TypeError("индекс должен быть целым числом, кортежем или списком")
        if len(multi_index) != self.ndim:
            raise IndexError(f"размерность индекса {len(multi_index)} не совпадает с порядком тензора {self.ndim}")
        for axis, index in enumerate(multi_index):
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("все индексы должны быть целыми числами")
            if index < 0 or index >= self.shape[axis]:
                raise IndexError("индекс выходит за границы")
        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(index, self.strides)
        return self.data[flat_index]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(index, self.strides)
        self.data[flat_index] = value

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        checked_shape = validate_shape(new_shape)
        if compute_size(checked_shape) != self.size:
            raise ValueError(f"нельзя изменить форму тензора размера {self.size} на форму {checked_shape}")
        return DenseTensor(checked_shape, data=self.data[:])

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if not isinstance(mode, int) or isinstance(mode, bool):
            raise TypeError("mode должен быть целым числом")
        if mode < 0 or mode >= self.ndim:
            raise ValueError("mode выходит за допустимые пределы")
        rest_shape = self.shape[:mode] + self.shape[mode + 1:]
        n_rows=self.shape[mode]
        n_cols=compute_size(rest_shape)
        rest_strides=compute_strides(rest_shape)
        result=DenseTensor.zeros((n_rows, n_cols))
        for flat_index,value in enumerate(self.data):
            full_index=flat_to_multi_index(flat_index,self.shape)
            row=full_index[mode]
            rest_index=full_index[:mode]+full_index[mode+ 1:]
            col=multi_index_to_flat(rest_index,rest_strides)
            result[row,col] = value
        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if not isinstance(k,int) or isinstance(k,bool):
            raise TypeError("k должен быть целым числом")
        if k< 0 or k>= self.ndim - 1:
            raise ValueError("k должно удовлетворять условию 0 <= k < ndim - 1")
        n_rows=compute_size(self.shape[:k+ 1])
        n_cols=compute_size(self.shape[k+ 1:])
        return DenseTensor((n_rows, n_cols), data=self.data[:])

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data[:])

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        return math.sqrt(sum(value * value for value in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        if not isinstance(other, DenseTensor):
            return NotImplemented
        check_shapes_match(self.shape, other.shape)
        data = [a + b for a,b in zip(self.data,other.data)]
        return DenseTensor(self.shape, data=data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        if not isinstance(other, DenseTensor):
            return NotImplemented
        check_shapes_match(self.shape, other.shape)
        data = [a-b for a,b in zip(self.data,other.data)]
        return DenseTensor(self.shape, data=data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        data = [value * scalar for value in self.data]
        return DenseTensor(self.shape, data=data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self * -1

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if not isinstance(other, DenseTensor):
            return False
        if self.shape != other.shape:
            return False
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        def build(axis: int, offset: int):
            if axis==self.ndim:
                return self.data[offset]
            return [
                build(axis+1,offset + i*self.strides[axis])
                for i in range(self.shape[axis])]
        return build(0, 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.to_nested_list()})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()
