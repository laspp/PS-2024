package main

import (
	"flag"
	"pozdrav-gpe/cudago"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

func main() {

	bPtr := flag.Int("b", 1, "num blocks")
	tPtr := flag.Int("t", 1, "num threads")
	flag.Parse()

	var err error

	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	gridSize := cuda.Dim3{X: uint32(*bPtr), Y: 1, Z: 1}
	blockSize := cuda.Dim3{X: uint32(*tPtr), Y: 1, Z: 1}

	var void *struct{}
	err = cudago.Hello(gridSize, blockSize, unsafe.Pointer(void))
	if err != nil {
		panic(err)
	}
}
