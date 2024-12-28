// računanje razlike vektorjev
// 		argumenti: število blokov, število niti in dolžina vektorjev
// 		elementi vektorjev so inicializirani naključno
// srun --reservation=fri --partition=gpu --gpus=1 go run main.go -b 0 -t 1024 -s 268435456
// srun --reservation=fri --partition=gpu --gpus=1 go run -gcflags '-N' main.go -b 0 -t 1024 -s 268435456

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"razdalja/cudago"
	"time"
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

	// izračunamo potrebno število blokov
	numBlocks := *numBlocksPtr
	if numBlocks == 0 {
		numBlocks = (*vectorSizePtr-1) / *numThreadsPtr + 1
	}

	var err error

	// inicializiramo napravo
	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	// prirpavimo dogodke za merjenje časa
	startDevice, err := cuda.NewEvent()
	if err != nil {
		panic(err)
	}
	stopDevice, err := cuda.NewEvent()
	if err != nil {
		panic(err)
	}

	// rezerviramo pomnilnik na gostitelju
	hs := make([]float32, 1)
	ha := make([]float32, *vectorSizePtr)
	hb := make([]float32, *vectorSizePtr)

	// rezerviramo pomnilnik na napravi
	sizeFloat32 := int(unsafe.Sizeof(float32(0.0)))
	sizeMem := uint64(*vectorSizePtr * sizeFloat32)

	ds, err := cuda.DeviceMemAlloc(uint64(sizeFloat32))
	if err != nil {
		panic(err)
	}
	defer ds.Free()
	da, err := cuda.DeviceMemAlloc(sizeMem)
	if err != nil {
		panic(err)
	}
	defer da.Free()
	db, err := cuda.DeviceMemAlloc(sizeMem)
	if err != nil {
		panic(err)
	}
	defer db.Free()

	// nastavimo vrednosti vektorjev a in b na gostitelju
	for i := 0; i < *vectorSizePtr; i++ {
		ha[i] = rand.Float32()
		hb[i] = rand.Float32()
	}

	// merjenje časa na napravi - začetek
	err = startDevice.Record(nil)
	if err != nil {
		panic(err)
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
	gridSize := cuda.Dim3{X: uint32(numBlocks), Y: 1, Z: 1}
	blockSize := cuda.Dim3{X: uint32(*numThreadsPtr), Y: 1, Z: 1}
	err = cudago.VectorDistance8aEx(gridSize, blockSize, uint64(*numThreadsPtr*sizeFloat32), nil, ds.Ptr, da.Ptr, db.Ptr, int32(*vectorSizePtr))
	if err != nil {
		panic(err)
	}

	// skalar s prekopiramo iz naprave na gostitelja
	err = ds.MemcpyFromDevice(uintptr(unsafe.Pointer(&hs[0])), uint64(sizeFloat32))

	// dokončamo izračun razdalje za napravo
	distDevice := math.Sqrt(float64(hs[0]))

	// merjenje časa na napravi - konec
	err = stopDevice.Record(nil)
	if err != nil {
		panic(err)
	}

	// počakamo, da se procesiranje zahtev na napravi zaključi
	err = cuda.CurrentContextSynchronize()
	if err != nil {
		panic(err)
	}

	// preverimo rezultat
	startHost := time.Now()
	distHost := float64(0.0)
	var diff float32
	for i := 0; i < *vectorSizePtr; i++ {
		diff = ha[i] - hb[i]
		distHost += float64(diff * diff)
	}
	distHost = math.Sqrt(distHost)
	timeHost := time.Since(startHost)

	// rezultata izpišemo
	timeDevice, err := cuda.EventElapsedTime(startDevice, stopDevice)
	if err != nil {
		panic(err)
	}
	fmt.Printf("naprava:      %f (%f ms)\ngostitelj:    %f (%f ms)\nnapaka (rel): %e\n",
		distDevice, timeDevice, distHost, float64(timeHost)/1e6, math.Abs(distDevice/distHost-1))

}
