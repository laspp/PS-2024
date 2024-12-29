// računanje razlike vektorjev
// 		argumenti: število blokov, število niti in dolžina vektorjev
// 		elementi vektorjev so inicializirani naključno
// nadgradimo računanje razlike vektorjev
// 		na napravi razliko elementov kvadriramo in shranimo v vektor c
//		na gostitelju seštejemo vse elemente vektorja c
//
// srun --reservation=fri --partition=gpu --gpus=1 go run urejanje-g.go -t 1024 -s 8388608

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

	// rezerviramo pomnilnik na gostitelju
	hc := make([]float32, *vectorSizePtr)
	ha := make([]float32, *vectorSizePtr)
	hb := make([]float32, *vectorSizePtr)

	// velikosti struktur v bajtih
	bytesFloat32 := int(unsafe.Sizeof(float32(0.0)))
	bytesVector := uint64(*vectorSizePtr * bytesFloat32)

	// rezerviramo pomnilnik na napravi
	dc, err := cuda.DeviceMemAlloc(bytesVector)
	if err != nil {
		panic(err)
	}
	defer dc.Free()
	da, err := cuda.DeviceMemAlloc(bytesVector)
	if err != nil {
		panic(err)
	}
	defer da.Free()
	db, err := cuda.DeviceMemAlloc(bytesVector)
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
	startDevice := time.Now()

	// prenesemo vektorja a in b iz gostitelja na napravo
	err = da.MemcpyToDevice(uintptr(unsafe.Pointer(&ha[0])), bytesVector)
	if err != nil {
		panic(err)
	}
	err = db.MemcpyToDevice(uintptr(unsafe.Pointer(&hb[0])), bytesVector)
	if err != nil {
		panic(err)
	}

	// merjenje časa izvajanja ščepca na napravi - začetek
	startKernel := time.Now()

	// zaženemo kodo na napravi
	gridSize := cuda.Dim3{X: uint32(numBlocks), Y: 1, Z: 1}
	blockSize := cuda.Dim3{X: uint32(*numThreadsPtr), Y: 1, Z: 1}
	err = cudago.VectorDistanceG(gridSize, blockSize, dc.Ptr, da.Ptr, db.Ptr, int32(*vectorSizePtr))
	if err != nil {
		panic(err)
	}

	// počakamo, da se procesiranje zahtev na napravi zaključi
	err = cuda.CurrentContextSynchronize()
	if err != nil {
		panic(err)
	}

	// merjenje časa izvajanja ščepca na napravi - konec
	timeKernel := time.Since(startKernel)

	// vektor c prekopiramo iz naprave na gostitelja
	err = dc.MemcpyFromDevice(uintptr(unsafe.Pointer(&hc[0])), bytesVector)

	// dokončamo izračun razdalje za napravo
	distDevice := float64(0.0)
	for i := 0; i < *vectorSizePtr; i++ {
		distDevice += float64(hc[i])
	}
	distDevice = math.Sqrt(distDevice)

	// merjenje časa na napravi - konec
	timeDevice := time.Since(startDevice)

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

	// izpišemo rezultate
	fmt.Printf("naprava:      %f (%v ms/%v us)\n", distDevice, timeDevice.Milliseconds(), timeKernel.Microseconds())
	fmt.Printf("gostitelj:    %f (%v ms)\n", distHost, timeHost.Milliseconds())
	fmt.Printf("napaka (rel): %e\n", math.Abs(distDevice/distHost-1))
}
