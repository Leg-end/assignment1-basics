try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping


def doc(s):
    if hasattr(s, '__call__'):
        s = s.__doc__

    def f(g):
        g.__doc__ = s
        return g
    return f


class HeapDictDescending(MutableMapping):
    __marker = object()

    def __init__(self, *args, **kw):
        self.heap = []
        self.d = {}
        self.update(*args, **kw)

    @doc(dict.clear)
    def clear(self):
        del self.heap[:]
        self.d.clear()

    @doc(dict.__setitem__)
    def __setitem__(self, key, value):
        if key in self.d:
            self.pop(key)
        wrapper = [value, key, len(self)]
        self.d[key] = wrapper
        self.heap.append(wrapper)
        self._increase_key(len(self.heap) - 1)  # 改为 _increase_key

    def _max_heapify(self, i):  # 改为 _max_heapify
        n = len(self.heap)
        h = self.heap
        while True:
            l = (i << 1) + 1  # 左子节点
            r = (i + 1) << 1  # 右子节点
            largest = i

            if l < n and h[l][0] > h[largest][0]:  # 改为 >（找最大值）
                largest = l
            if r < n and h[r][0] > h[largest][0]:  # 改为 >
                largest = r

            if largest == i:
                break

            self._swap(i, largest)
            i = largest

    def _increase_key(self, i):  # 改为 _increase_key
        while i > 0:
            parent = (i - 1) >> 1  # 父节点
            if self.heap[parent][0] > self.heap[i][0]:  # 改为 >
                break
            self._swap(i, parent)
            i = parent

    def _swap(self, i, j):
        h = self.heap
        h[i], h[j] = h[j], h[i]
        h[i][2] = i
        h[j][2] = j

    @doc(dict.__delitem__)
    def __delitem__(self, key):
        wrapper = self.d[key]
        while wrapper[2] > 0:  # 确保不是根节点
            parent = (wrapper[2] - 1) >> 1
            self._swap(wrapper[2], parent)
        self.popitem()

    @doc(dict.__getitem__)
    def __getitem__(self, key):
        return self.d[key][0]

    @doc(dict.__iter__)
    def __iter__(self):
        return iter(self.d)

    def popitem(self):
        """Remove and return the (key, value) pair with the highest value."""
        if not self.heap:
            raise KeyError("popitem(): dictionary is empty")
        
        wrapper = self.heap[0]
        if len(self.heap) == 1:
            self.heap.pop()
        else:
            self.heap[0] = self.heap.pop()
            self.heap[0][2] = 0
            self._max_heapify(0)  # 改为 _max_heapify
        
        del self.d[wrapper[1]]
        return wrapper[1], wrapper[0]

    @doc(dict.__len__)
    def __len__(self):
        return len(self.d)

    def peekitem(self):
        """Return the (key, value) pair with the highest value without removing it."""
        if not self.heap:
            raise KeyError("peekitem(): dictionary is empty")
        return (self.heap[0][1], self.heap[0][0])