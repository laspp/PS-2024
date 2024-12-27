package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"razlika/cudago"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

func main() {

	// preberemo argumente iz ukazne vrstice
	bPtr := flag.Int("b", 1, "num blocks")
	tPtr := flag.Int("t", 1, "num threads")
	sPtr := flag.Int("s", 1, "vector size")
	flag.Parse()

	var err error

	// inicializiramo napravo
	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	// rezerviramo pomnilnik na gostitelju
	hc := make([]float32, *sPtr)
	ha := make([]float32, *sPtr)
	hb := make([]float32, *sPtr)

	// rezerviramo pomnilnik na napravi
	memSize := *sPtr * int(unsafe.Sizeof(float32(0.0)))
	dc, err := cuda.DeviceMemAlloc(uint64(*sPtr * memSize))
	if err != nil {
		panic(err)
	}
	da, err := cuda.DeviceMemAlloc(uint64(*sPtr * memSize))
	if err != nil {
		panic(err)
	}
	db, err := cuda.DeviceMemAlloc(uint64(*sPtr * memSize))
	if err != nil {
		panic(err)
	}

	// nastavimo vrednosti vektorjev a in b na gostitelju
	for i := 0; i < *sPtr; i++ {
		ha[i] = rand.Float32()
		hb[i] = rand.Float32()
	}

	// prenesemo vektorja a in b iz gostitelja na napravo
	err = da.MemcpyToDevice(uintptr(unsafe.Pointer(&ha[0])), uint64(memSize))
	if err != nil {
		panic(err)
	}
	err = db.MemcpyToDevice(uintptr(unsafe.Pointer(&hb[0])), uint64(memSize))
	if err != nil {
		panic(err)
	}

	// zaženemo kodo na napravi
	gridSize := cuda.Dim3{X: uint32(*bPtr), Y: 1, Z: 1}
	blockSize := cuda.Dim3{X: uint32(*tPtr), Y: 1, Z: 1}
	err = cudago.VectorSubtract(gridSize, blockSize, dc.Ptr, da.Ptr, db.Ptr, int32(*sPtr))
	if err != nil {
		panic(err)
	}

	// počakamo, da vse niti na napravi zaključijo
	err = cuda.CurrentContextSynchronize()
	if err != nil {
		panic(err)
	}

	// vektor c prekopiramo iz naprave na gostitelja
	err = dc.MemcpyFromDevice(uintptr(unsafe.Pointer(&hc[0])), uint64(memSize))

	// preverimo rezultat
	ok := true
	for i := 0; i < *sPtr; i++ {
		if ha[i]-hb[i] != hc[i] {
			ok = false
		}
	}
	if ok {
		fmt.Println("Result is correct.")
	} else {
		fmt.Println("Result is wrong.")
	}
}
