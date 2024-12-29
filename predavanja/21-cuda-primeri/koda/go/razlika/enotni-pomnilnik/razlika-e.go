// računanje razlike vektorjev
// 		argumenti: število blokov, število niti in dolžina vektorjev
// 		elementi vektorjev so inicializirani naključno
// rešitev z uporabo enotnega pomnilnika

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
	numBlocksPtr := flag.Int("b", 1, "num blocks")
	numThreadsPtr := flag.Int("t", 1, "num threads")
	vectorSizePtr := flag.Int("s", 1, "vector size")
	flag.Parse()
	if *numBlocksPtr < 0 || *numThreadsPtr <= 0 || *vectorSizePtr <= 0 {
		panic("Wrong arguments")
	}

	var err error

	// inicializiramo napravo
	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	// rezerviramo pomnilnik
	sizeFloat32 := uint64(unsafe.Sizeof(float32(0.0)))
	c, err := cuda.ManagedMemAlloc[float32](uint64(*vectorSizePtr), sizeFloat32)
	if err != nil {
		panic(err)
	}
	defer c.Free()
	a, err := cuda.ManagedMemAlloc[float32](uint64(*vectorSizePtr), sizeFloat32)
	if err != nil {
		panic(err)
	}
	defer a.Free()
	b, err := cuda.ManagedMemAlloc[float32](uint64(*vectorSizePtr), sizeFloat32)
	if err != nil {
		panic(err)
	}
	defer b.Free()

	// nastavimo vrednosti vektorjev a in b na gostitelju
	for i := 0; i < *vectorSizePtr; i++ {
		a.Arr[i] = rand.Float32()
		b.Arr[i] = rand.Float32()
	}

	// zaženemo kodo na napravi
	numBlocks := *numBlocksPtr
	if numBlocks == 0 {
		numBlocks = (*vectorSizePtr-1) / *numThreadsPtr + 1
	}
	gridSize := cuda.Dim3{X: uint32(numBlocks), Y: 1, Z: 1}
	blockSize := cuda.Dim3{X: uint32(*numThreadsPtr), Y: 1, Z: 1}
	err = cudago.VectorSubtract4(gridSize, blockSize, c.Ptr, a.Ptr, b.Ptr, int32(*vectorSizePtr))
	if err != nil {
		panic(err)
	}

	// počakamo, da se procesiranje zahtev na napravi zaključi
	err = cuda.CurrentContextSynchronize()
	if err != nil {
		panic(err)
	}

	// preverimo rezultat
	ok := true
	for i := 0; i < *vectorSizePtr; i++ {
		if a.Arr[i]-b.Arr[i] != c.Arr[i] {
			ok = false
		}
	}
	if ok {
		fmt.Println("Result is correct.")
	} else {
		fmt.Println("Result is wrong.")
	}
}
