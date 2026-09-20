package main

import "testing"

// Two positions that land in the same slot must not read each other's value.
// The table previously stored no key, so the second lookup here returned the
// first position's evaluation and backed it up the tree as if it were real.
func TestTableDistinguishesCollidingKeys(t *testing.T) {
	tt := NewTable(1 << 16)
	const a uint64 = 0x1234_5678_0000_ABCD
	b := a + 1<<16 // same low 16 bits, different position

	tt.Store(a, 0.75)
	if v, ok := tt.Get(a); !ok || v != 0.75 {
		t.Fatalf("Get(a) = (%v, %v), want (0.75, true)", v, ok)
	}
	if v, ok := tt.Get(b); ok {
		t.Errorf("Get(b) = (%v, true) from a's slot; colliding keys must miss", v)
	}

	tt.Store(b, -0.25)
	if v, ok := tt.Get(b); !ok || v != -0.25 {
		t.Errorf("Get(b) = (%v, %v), want (-0.25, true)", v, ok)
	}
	if _, ok := tt.Get(a); ok {
		t.Error("a should have been evicted by b, replace-always")
	}
}

// A zero hash is a legitimate key, not an empty slot. Without the filled bit an
// untouched slot would answer Get(0) with a confident 0.0.
func TestTableZeroHashIsNotAnEmptySlot(t *testing.T) {
	tt := NewTable(256)
	if _, ok := tt.Get(0); ok {
		t.Fatal("empty table answered Get(0)")
	}
	tt.Store(0, 0.5)
	if v, ok := tt.Get(0); !ok || v != 0.5 {
		t.Errorf("Get(0) = (%v, %v), want (0.5, true)", v, ok)
	}
}

// Sizes that are not powers of two must still mask correctly; a mask of n-1 on
// a non-power-of-two silently drops slots and aliases keys.
func TestTableRoundsSizeToPowerOfTwo(t *testing.T) {
	tt := NewTable(1000)
	if got := len(tt.slots); got != 1024 {
		t.Fatalf("NewTable(1000) has %d slots, want 1024", got)
	}
	if tt.mask != 1023 {
		t.Fatalf("mask = %d, want 1023", tt.mask)
	}
	for _, size := range []int{0, 1, 300} {
		s := NewTable(size)
		if n := len(s.slots); n&(n-1) != 0 {
			t.Errorf("NewTable(%d) has %d slots, not a power of two", size, n)
		}
	}
}
