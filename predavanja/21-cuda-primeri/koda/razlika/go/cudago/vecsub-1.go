// Code generated by cudago. Edit at your own risk.
package cudago

import (
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

// here just to force usage of unsafe package
var __vecsub_1_useless_var__ unsafe.Pointer = nil

const (
	KeyVecsub_1 = "vecsub_1"
)

type vectorsubtract1Args struct {
	c   uintptr
	a   uintptr
	b   uintptr
	len int32
}

/*var (
    vectorsubtract1Args = vectorsubtract1Args{}

)*/

func VectorSubtract1(grid, block cuda.Dim3, c uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_vecsub_1()
	if err != nil {
		return err
	}
	kern, err := getKernel("vecsub_1", "vectorSubtract1")
	if err != nil {
		return err
	}

	params := vectorsubtract1Args{
		c:   c,
		a:   a,
		b:   b,
		len: len,
	}

	return kern.Launch(grid, block, unsafe.Pointer(&params.c), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}

func VectorSubtract1Ex(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, c uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_vecsub_1()
	if err != nil {
		return err
	}
	kern, err := getKernel("vecsub_1", "vectorSubtract1")
	if err != nil {
		return err
	}

	params := vectorsubtract1Args{
		c:   c,
		a:   a,
		b:   b,
		len: len,
	}

	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.c), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}

var loaded_vecsub_1 = false

var pathToCompile_vecsub_1 = "./vecsub-1.cu"

func autoloadLib_vecsub_1() error {
	var code []byte
	if loaded_vecsub_1 {
		return nil
	}
	code, err := compileFile(pathToCompile_vecsub_1)
	if err != nil {
		return err
	}
	err = InitLibrary([]byte(code), "vecsub_1")
	if err != nil {
		return err
	}
	loaded_vecsub_1 = true
	return nil
}
