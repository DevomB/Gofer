package main

// slot is one transposition record. The full Zobrist key is kept alongside the
// value: indexing by hash&mask alone cannot tell two positions apart once they
// land in the same slot, and serving one position's evaluation for another is
// silent - it backs a wrong value up the tree and looks like a bad evaluator.
type slot struct {
	key    uint64
	value  float64
	filled bool
}

// Table is a Zobrist-keyed transposition table with replace-always eviction.
type Table struct {
	slots []slot
	mask  uint64
}

// NewTable creates a table with the given slot count, rounded up to a power of
// two so the index is a mask rather than a division.
func NewTable(size int) *Table {
	n := 256
	for n < size {
		n <<= 1
	}
	return &Table{slots: make([]slot, n), mask: uint64(n - 1)}
}

// Get returns the value stored for hash, if this exact position is in the table.
func (t *Table) Get(hash uint64) (float64, bool) {
	s := t.slots[hash&t.mask]
	if !s.filled || s.key != hash {
		return 0, false
	}
	return s.value, true
}

// Store records value for hash, evicting whatever shared the slot.
func (t *Table) Store(hash uint64, value float64) {
	t.slots[hash&t.mask] = slot{key: hash, value: value, filled: true}
}
