// Code generated by cudago. Edit at your own risk.
package cudago

import (
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

// here just to force usage of unsafe package
var __vecsub_2_useless_var__ unsafe.Pointer = nil

const (
	KeyVecsub_2 = "vecsub_2"
)

type vectorsubtract2Args struct {
	c   uintptr
	a   uintptr
	b   uintptr
	len int32
}

/*var (
    vectorsubtract2Args = vectorsubtract2Args{}

)*/

func VectorSubtract2(grid, block cuda.Dim3, c uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_vecsub_2()
	if err != nil {
		return err
	}
	kern, err := getKernel("vecsub_2", "vectorSubtract2")
	if err != nil {
		return err
	}

	params := vectorsubtract2Args{
		c:   c,
		a:   a,
		b:   b,
		len: len,
	}

	return kern.Launch(grid, block, unsafe.Pointer(&params.c), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}

func VectorSubtract2Ex(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, c uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_vecsub_2()
	if err != nil {
		return err
	}
	kern, err := getKernel("vecsub_2", "vectorSubtract2")
	if err != nil {
		return err
	}

	params := vectorsubtract2Args{
		c:   c,
		a:   a,
		b:   b,
		len: len,
	}

	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.c), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}

var loaded_vecsub_2 = false

var pathToCompile_vecsub_2 = "./vecsub-2.cu"

func autoloadLib_vecsub_2() error {
	var code []byte
	if loaded_vecsub_2 {
		return nil
	}
	code, err := compileFile(pathToCompile_vecsub_2)
	if err != nil {
		return err
	}
	err = InitLibrary([]byte(code), "vecsub_2")
	if err != nil {
		return err
	}
	loaded_vecsub_2 = true
	return nil
}
