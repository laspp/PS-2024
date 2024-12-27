// računanje razlike vektorjev
// 		argumenti: število blokov, število niti in dolžina vektorjev
// 		elementi vektorjev so inicializirani naključno

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

	// rezerviramo pomnilnik na gostitelju
	hc := make([]float32, *vectorSizePtr)
	ha := make([]float32, *vectorSizePtr)
	hb := make([]float32, *vectorSizePtr)

	// rezerviramo pomnilnik na napravi
	sizeFloat32 := int(unsafe.Sizeof(float32(0.0)))
	sizeMem := uint64(*vectorSizePtr * sizeFloat32)

	dc, err := cuda.DeviceMemAlloc(sizeMem)
	if err != nil {
		panic(err)
	}
	da, err := cuda.DeviceMemAlloc(sizeMem)
	if err != nil {
		panic(err)
	}
	db, err := cuda.DeviceMemAlloc(sizeMem)
	if err != nil {
		panic(err)
	}

	// nastavimo vrednosti vektorjev a in b na gostitelju
	for i := 0; i < *vectorSizePtr; i++ {
		ha[i] = rand.Float32()
		hb[i] = rand.Float32()
	}

	// prenesemo vektorja a in b iz gostitelja na napravo
	err = da.MemcpyToDevice(uintptr(unsafe.Pointer(&ha[0])), sizeMem)
	if err != nil {
		panic(err)
	}
	err = db.MemcpyToDevice(uintptr(unsafe.Pointer(&hb[0])), sizeMem)
	if err != nil {
		panic(err)
	}

	// zaženemo kodo na napravi
	numBlocks := *numBlocksPtr
	if numBlocks == 0 {
		numBlocks = (*vectorSizePtr-1) / *numThreadsPtr + 1
	}
	gridSize := cuda.Dim3{X: uint32(numBlocks), Y: 1, Z: 1}
	blockSize := cuda.Dim3{X: uint32(*numThreadsPtr), Y: 1, Z: 1}
	err = cudago.VectorSubtract(gridSize, blockSize, dc.Ptr, da.Ptr, db.Ptr, int32(*vectorSizePtr))
	if err != nil {
		panic(err)
	}

	// počakamo, da vse niti na napravi zaključijo
	err = cuda.CurrentContextSynchronize()
	if err != nil {
		panic(err)
	}

	// vektor c prekopiramo iz naprave na gostitelja
	err = dc.MemcpyFromDevice(uintptr(unsafe.Pointer(&hc[0])), sizeMem)

	// preverimo rezultat
	ok := true
	for i := 0; i < *vectorSizePtr; i++ {
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
