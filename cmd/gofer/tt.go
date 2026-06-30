package main

// Entry is a transposition table record.
type Entry struct {
	Depth int
	Value float64
}

// Table is a Zobrist-keyed transposition table.
type Table struct {
	slots []Entry
	mask  uint64
}

// NewTable creates a TT with the given slot count (power of two).
func NewTable(size int) *Table {
	if size < 256 {
		size = 256
	}
	return &Table{
		slots: make([]Entry, size),
		mask:  uint64(size - 1),
	}
}

// Get looks up hash.
func (t *Table) Get(hash uint64) (Entry, bool) {
	e := t.slots[hash&t.mask]
	if e.Depth == 0 {
		return Entry{}, false
	}
	return e, true
}

// Store saves an entry (replace always).
// Replace-always eviction; no depth preference.
func (t *Table) Store(hash uint64, e Entry) {
	t.slots[hash&t.mask] = e
}
