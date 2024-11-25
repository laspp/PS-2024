package vector_clock

import "fmt"

// VectorClock is a map of string to int
type VectorClock map[string]int

// New returns a new VectorClock
func New() VectorClock {
	return make(VectorClock)
}

// Increment the clock for the given id
func (vc VectorClock) Increment(id string) {
	vc[id]++
}

// Set the clock for the given id
func (vc VectorClock) Set(id string, value int) {
	vc[id] = value
}

// Get the clock for the given id
func (vc VectorClock) Get(id string) int {
	return vc[id]
}

// Merge the given VectorClock into this one
func (vc VectorClock) Merge(other VectorClock) {
	for id, value := range other {
		if value > vc[id] {
			vc[id] = value
		}
	}
}

// Equal returns true if the given VectorClock is equal to this one
func (vc VectorClock) Equal(other VectorClock) bool {
	for id, value := range vc {
		if other[id] != value {
			return false
		}
	}
	return true
}

// Compare returns -1 if this VectorClock is less than the given one, 1 if it is greater, and 0 if they are concurrent
func (vc VectorClock) Compare(other VectorClock) int {
	less, greater := false, false
	for id, value := range vc {
		if other[id] < value {
			less = true
		} else if other[id] > value {
			greater = true
		}
	}
	if less && greater {
		return 0
	} else if less {
		return -1
	} else if greater {
		return 1
	}
	return 0
}

// String returns a string representation of this VectorClock
func (vc VectorClock) String() string {
	return fmt.Sprintf("%v", map[string]int(vc))
}

// Copy returns a copy of this VectorClock
func (vc VectorClock) Copy() VectorClock {
	copiedVClock := New()
	copiedVClock.Merge(vc)
	return copiedVClock
}
