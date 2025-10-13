try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
    
# adapt from https://github.com/DanielStutzbach/heapdict/tree/master


def doc(s):
    if hasattr(s, '__call__'):
        s = s.__doc__

    def f(g):
        g.__doc__ = s
        return g
    return f


class heapdict(MutableMapping):
    __slots__ = ('heap', 'd')
    __marker = object()

    def __init__(self, *args, **kw):
        self.heap = []
        self.d = {}
        self.update(*args, **kw)

    @doc(dict.clear)
    def clear(self):
        """D.clear() -> None.  Remove all items from D."""
        self.heap.clear()
        self.d.clear()

    @doc(dict.__setitem__)
    def __setitem__(self, key, value):
        if key in self.d:
            del self[key]
        wrapper = [value, key, len(self.heap)]
        self.d[key] = wrapper
        self.heap.append(wrapper)
        self._increase_key(len(self.heap) - 1)

    def _max_heapify(self, i):
        heap = self.heap
        n = len(heap)
        
        while True:
            left = (i << 1) + 1
            right = left + 1
            largest = i

            if left < n and heap[left][0] > heap[largest][0]:
                largest = left
            if right < n and heap[right][0] > heap[largest][0]:
                largest = right

            if largest == i:
                break

            self._swap(i, largest)
            i = largest

    def _increase_key(self, i):
        heap = self.heap
        while i > 0:
            parent = (i - 1) >> 1
            if heap[i][0] <= heap[parent][0]:
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
        if key not in self.d:
            raise KeyError(key)
            
        wrapper = self.d[key]
        pos = wrapper[2]
        
        # 将要删除的元素交换到堆顶
        while pos > 0:
            parent = (pos - 1) >> 1
            self._swap(pos, parent)
            pos = parent
            
        self.popitem()

    @doc(dict.__getitem__)
    def __getitem__(self, key):
        return self.d[key][0]

    @doc(dict.__iter__)
    def __iter__(self):
        return iter(self.d)

    def popitem(self):
        """D.popitem() -> (k, v), remove and return the (key, value) pair with highest value."""
        if not self.heap:
            raise KeyError("popitem from empty heapdict")
            
        wrapper = self.heap[0]
        last = self.heap.pop()
        
        if self.heap:
            self.heap[0] = last
            last[2] = 0
            self._max_heapify(0)
            
        del self.d[wrapper[1]]
        return wrapper[1], wrapper[0]

    @doc(dict.__len__)
    def __len__(self):
        return len(self.d)

    def peekitem(self):
        """D.peekitem() -> (k, v), return the (key, value) pair with highest value."""
        if not self.heap:
            raise KeyError("peekitem from empty heapdict")
        return self.heap[0][1], self.heap[0][0]

    def __repr__(self):
        return f"heapdict({dict(self)})"